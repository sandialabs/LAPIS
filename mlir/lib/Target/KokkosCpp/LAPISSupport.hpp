#include <Kokkos_Core.hpp>
#include <type_traits>
#include <cstdint>
#include <unistd.h>
#include <iostream>

template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

namespace LAPIS
{
  using TeamPolicy = Kokkos::TeamPolicy<>;
  using TeamMember = typename TeamPolicy::member_type;

  template<typename V>
  StridedMemRefType<typename V::value_type, V::rank> viewToStridedMemref(const V& v)
  {
    StridedMemRefType<typename V::value_type, V::rank> smr;
    smr.basePtr = v.data();
    smr.data = v.data();
    smr.offset = 0;
    for(int i = 0; i < int(V::rank); i++)
    {
      smr.sizes[i] = v.extent(i);
      smr.strides[i] = v.stride(i);
    }
    return smr;
  }

  template<typename V>
  V stridedMemrefToView(const StridedMemRefType<typename V::value_type, V::rank>& smr)
  {
    using Layout = typename V::array_layout;
    static_assert(std::is_same_v<typename V::memory_space, Kokkos::HostSpace> ||
        std::is_same_v<typename V::memory_space, Kokkos::AnonymousSpace>,
        "Can only convert a StridedMemRefType to a Kokkos::View in HostSpace.");
    if constexpr(std::is_same_v<Layout, Kokkos::LayoutStride>)
    {
      size_t extents[8] = {0};
      size_t strides[8] = {0};
      for(int i = 0; i < V::rank; i++) {
        extents[i] = smr.sizes[i];
        strides[i] = smr.strides[i];
      }
      Layout layout(
          extents[0], strides[0],
          extents[1], strides[1],
          extents[2], strides[2],
          extents[3], strides[3],
          extents[4], strides[4],
          extents[5], strides[5],
          extents[6], strides[6],
          extents[7], strides[7]);
      return V(&smr.data[smr.offset], layout);
    }
    size_t extents[8] = {
      KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX,
      KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX, KOKKOS_INVALID_INDEX};
    for(int i = 0; i < V::rank; i++)
      extents[i] = smr.sizes[i];
    Layout layout(
        extents[0], extents[1], extents[2], extents[3],
        extents[4], extents[5], extents[6], extents[7]);
    if constexpr(std::is_same_v<Layout, Kokkos::LayoutLeft>)
    {
      int64_t expectedStride = 1;
      for(int i = 0; i < int(V::rank); i++)
      {
        if(expectedStride != smr.strides[i])
          Kokkos::abort("Cannot shallow-copy StridedMemRefType that is not contiguous and LayoutLeft to LayoutLeft Kokkos::View");
        expectedStride *= smr.sizes[i];
      }
    }
    else if constexpr(std::is_same_v<Layout, Kokkos::LayoutRight>)
    {
      int64_t expectedStride = 1;
      for(int i = int(V::rank) - 1; i >= 0; i--)
      {
        if(expectedStride != smr.strides[i])
          Kokkos::abort("Cannot shallow-copy StridedMemRefType that is not contiguous and LayoutRight to LayoutRight Kokkos::View");
        expectedStride *= smr.sizes[i];
      }
    }
    return V(&smr.data[smr.offset], layout);
  }

  // KeepAlive structure keeps a reference to Kokkos::Views which
  // are returned to Python. Since it's difficult to transfer ownership of a
  // Kokkos::View's memory to numpy, we just have the Kokkos::View maintain ownership
  // and return an unmanaged numpy array to Python.
  //
  // All these views will be deallocated during lapis_finalize to avoid leaking.
  // The downside is that if a function is called many times,
  // all its results are kept in memory at the same time.
  struct KeepAlive
  {
    virtual ~KeepAlive() {}
  };

  template<typename T>
  struct KeepAliveT : public KeepAlive
  {
    // Make a shallow-copy of val
    KeepAliveT(const T& val) : p(new T(val)) {}
    std::unique_ptr<T> p;
  };

  static std::vector<std::unique_ptr<KeepAlive>> alives;

  template<typename T>
  void keepAlive(const T& val)
  {
    alives.emplace_back(new KeepAliveT(val));
  }

  // DualView design
  // - DualView is a shallow object with a shared_ptr to a DualViewImpl.
  // - DualViewImpl has the actual host and device views as members
  //   - These may be managed or unmanaged
  // - DualViewImpl also has a shared_ptr reference to its "parent". This is another DualViewImpl (possibly of different type)
  //   that is considered the owner of the memory. The shared_ptr reference ensures
  //   - it stays alive as long as any child DualView is alive, even if the original parent declaration goes out of scope
  //   - it is deallocated as soon as the last child DualView goes out of scope
  // - Assume that any DualView's parent is contiguous, and can be deep-copied between h and d
  // - All DualViews with the same parent share the parent's modify flags
  //
  //  DualViewBase can also "keepAliveHost" to keep its host view alive until lapis_finalize is called.
  //  This is used to safely return host views to python for numpy arrays to alias.

  struct DualViewBase
  {
    enum AliasStatus
    {
        ALIAS_STATUS_UNKNOWN = 0,
        HOST_IS_ALIAS = 1,
        DEVICE_IS_ALIAS = 2,
        NEITHER_IS_ALIAS = 3
    };

    virtual ~DualViewBase() {}
    virtual void syncHost() = 0;
    virtual void syncDevice() = 0;
    virtual void keepAliveHost() = 0;
    bool modified_host = false;
    bool modified_device = false;
    std::shared_ptr<DualViewBase> parent;
    AliasStatus alias_status;

    void setParent(const std::shared_ptr<DualViewBase>& parent_)
    {
      this->parent = parent_;
    }
  };

  template<typename DataType, typename Layout>
    struct DualViewImpl : public DualViewBase
  {
    using HostView = Kokkos::View<DataType, Layout, Kokkos::DefaultHostExecutionSpace>;
    using DeviceView = Kokkos::View<DataType, Layout, Kokkos::DefaultExecutionSpace>;

    static constexpr bool deviceAccessesHost = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;
    static constexpr bool hostAccessesDevice = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;

    // Default constructor makes empty/non-allocated views
    DualViewImpl() : device_view(), host_view() {}

    // Constructor for allocating a new view.
    // Does not actually allocate anything yet; instead 
    DualViewImpl(
        const std::string& label,
        size_t ex0 = KOKKOS_INVALID_INDEX, size_t ex1 = KOKKOS_INVALID_INDEX, size_t ex2 = KOKKOS_INVALID_INDEX, size_t ex3 = KOKKOS_INVALID_INDEX,
        size_t ex4 = KOKKOS_INVALID_INDEX, size_t ex5 = KOKKOS_INVALID_INDEX, size_t ex6 = KOKKOS_INVALID_INDEX, size_t ex7 = KOKKOS_INVALID_INDEX)
    {
      if constexpr(hostAccessesDevice) {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_dev"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        host_view = HostView(device_view.data(), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
      else if constexpr(deviceAccessesHost) {
        // Otherwise, host_view must be a separate allocation.
        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_host"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        device_view = DeviceView(host_view.data(), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
      else {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_dev"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_host"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
    }

    // Constructor which is given explicit device and host views, and a parent.
    // This can be used for subviewing/casting operations.
    // Note: d,h should alias parent's memory, but they can
    // have a different data type and layout.
    DualViewImpl(DeviceView d, HostView h)
      : device_view(d), host_view(h) {}

    // Constructor taking a host or device view
    template<typename DT, typename... Args>
    DualViewImpl(Kokkos::View<DT, Args...> v)
    {
      using ViewType = decltype(v);
      using Space = typename ViewType::memory_space;
      if constexpr(std::is_same_v<typename DeviceView::memory_space, Space>) {
        // Treat v like a device view, even though it's possible that DeviceView and HostView have the same type.
        // In this case, the host view will alias it.
        modified_device = true;
        if constexpr(deviceAccessesHost) {
          host_view = HostView(v.data(), v.layout());
          alias_status = AliasStatus::HOST_IS_ALIAS;
        }
        else {
          host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, v.label() + "_host"), v.layout());
          alias_status = AliasStatus::NEITHER_IS_ALIAS;
        }
        device_view = v;
      }
      else {
        modified_host = true;
        if constexpr(deviceAccessesHost) {
          device_view = DeviceView(v.data(), v.layout());
          alias_status = AliasStatus::DEVICE_IS_ALIAS;
        }
        else {
          device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, v.label() + "_dev"), v.layout());
          alias_status = AliasStatus::NEITHER_IS_ALIAS;
        }
        host_view = v;
      }
    }

    void modifyHost()
    {
      parent->modified_host = true;
    }

    void modifyDevice()
    {
      parent->modified_device = true;
    }

    bool modifiedHost()
    {
      // note: parent may just point to this
      return parent->modified_host;
    }

    bool modifiedDevice()
    {
      // note: parent may just point to this
      return parent->modified_device;
    }

    void syncHost() override
    {
      if (device_view.data() == host_view.data()) {
        // Imitating Kokkos::DualView behavior: if device and host are the same space
        // then this sync (if required) is equivalent to a fence.
        if(parent->modified_device) {
          parent->modified_device = false;
          Kokkos::fence();
        }
      }
      else if (parent->modified_device) {
        if(parent.get() == this) {
          Kokkos::deep_copy(host_view, device_view);
          modified_device = false;
        }
        else {
          parent->syncHost();
        }
      }
    }

    void syncDevice() override
    {
      // If host and device views are the same, do not sync or fence
      // because all host execution spaces are synchronous.
      // Any changes on the host side are immediately visible on the device side.
      if (device_view.data() != host_view.data()) {
        if(parent.get() == this) {
          if(modified_host) {
            Kokkos::deep_copy(device_view, host_view);
            modified_host = false;
          }
        }
        else {
          parent->syncDevice();
        }
      }
    }

    void keepAliveHost() override
    {
      // keep the parent's host view alive.
      // It is assumed to be either managed,
      // or unmanaged but references memory (e.g. from numpy)
      // with a longer lifetime that any result from the current LAPIS function.
      //
      // However, if it's unmanaged because of aliasing during initialization,
      // then keep alive the device_view instead to avoid reference counting
      // issues in Kokkos::View.
      if(alias_status != AliasStatus::HOST_IS_ALIAS)
      {
        keepAlive(host_view);
      }else{
        keepAlive(device_view);
      }
    }

    void deallocate() {
      device_view = DeviceView();
      host_view = HostView();
    }

    size_t extent(int dim) {
      return device_view.extent(dim);
    }

    size_t stride(int dim) {
      return device_view.stride(dim);
    }

    DeviceView device_view;
    HostView host_view;
  };

  template<typename DataType, typename Layout>
  struct DualView
  {
    using ImplType = DualViewImpl<DataType, Layout>;
    using DeviceView = typename ImplType::DeviceView;
    using HostView = typename ImplType::HostView;

    std::shared_ptr<ImplType> impl;
    bool syncHostWhenDestroyed = false;

    DualView() {
      impl = std::make_shared<ImplType>();
      // Even though no data is allocated, set impl's parent to itself
      // so that sync/modify calls are well defined
      impl->setParent(impl);
    }

    DualView(
        const std::string& label,
        size_t ex0 = KOKKOS_INVALID_INDEX, size_t ex1 = KOKKOS_INVALID_INDEX, size_t ex2 = KOKKOS_INVALID_INDEX, size_t ex3 = KOKKOS_INVALID_INDEX,
        size_t ex4 = KOKKOS_INVALID_INDEX, size_t ex5 = KOKKOS_INVALID_INDEX, size_t ex6 = KOKKOS_INVALID_INDEX, size_t ex7 = KOKKOS_INVALID_INDEX) {
      impl = std::make_shared<ImplType>(label, ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      impl->setParent(impl);
    }

    template<typename V>
    DualView(const V& v) {
      static_assert(std::is_same_v<typename V::data_type, DataType>,
          "DualView constructor from view: data type must match exactly");
      impl = std::make_shared<ImplType>(v);
      impl->setParent(impl);
    }

    template<typename Parent>
    DualView(const DeviceView& d, const HostView& h, const Parent& parent) {
      impl = std::make_shared<ImplType>(d, h);
      // From the caller's point of view, parent is some DualView that is considered the parent of this.
      // But we need parent to point to the top-level parent, not just the immediate parent.
      // This way we don't have to pointer hop multiple times during sync/modify calls.
      impl->setParent(parent.impl->parent);
    }

    ~DualView() {
      if(syncHostWhenDestroyed) syncHost();
      DualViewBase* parent = impl->parent.get();
      impl.reset();
      // All DualViewBases keep a shared reference to themselves, so
      // parent always keeps a shared_ptr to itself. This would normally
      // prevent the parent destructor ever being called.
      //
      // So if parent now has a use count of 1, it is
      // only pointing to itself and no other DualView objects
      // are using it anymore.
      if(parent->parent.use_count() == 1) {
        // This will reset the last shared_ptr reference,
        // causing parent to be destroyed.
        parent->parent.reset();
      }
    }

    DeviceView device_view() const {
      return impl->device_view;
    }

    HostView host_view() const {
      return impl->host_view;
    }

    void modifyHost() {
      impl->parent->modified_host = true;
    }

    void modifyDevice() {
      impl->parent->modified_device = true;
    }

    bool modifiedHost() const {
      // note: parent may just point to this
      return impl->parent->modified_host;
    }

    bool modifiedDevice() const {
      // note: parent may just point to this
      return impl->parent->modified_device;
    }

    void syncHost() {
      impl->syncHost();
    }

    void syncDevice() {
      impl->syncDevice();
    }

    void deallocate() {
      // Default destructor of this will release reference to impl,
      // but this can also be used to explicitly release.
      impl.reset();
    }

    size_t extent(int dim) const {
      return impl->extent(dim);
    }

    size_t stride(int dim) const {
      return impl->stride(dim);
    }

    void keepAliveHost() const {
      impl->parent->keepAliveHost();
    }

    void syncHostOnDestroy() {
      syncHostWhenDestroyed = true;
    }
  };

  inline int threadParallelVectorLength(int par) {
    if (par < 1)
      return 1;
    int max_vector_length = TeamPolicy::vector_length_max();
    int vector_length = 1;
    while(vector_length < max_vector_length && vector_length * 6 < par) vector_length *= 2;
    return vector_length;
  }
} // namespace LAPIS

