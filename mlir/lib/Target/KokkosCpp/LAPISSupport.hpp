#include <Kokkos_Core.hpp>
#include <type_traits>
#include <cstdint>
#include <unistd.h>
#include <iostream>

// If building a CPP driver, we can use the original StridedMemRefType class from MLIR,
// so do not redefine it here.
#ifndef PYTACO_CPP_DRIVER
template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};
#endif

// If building a CPP driver, need to provide a version of
// _mlir_ciface_newSparseTensor() that takes underlying integer types, not enum types like DimLevelType.
// The MLIR-Kokkos generated code doesn't know about the enum types at all.
#ifdef PYTACO_CPP_DRIVER
int8_t* _mlir_ciface_newSparseTensor(
  StridedMemRefType<index_type, 1> *dimSizesRef,
  StridedMemRefType<index_type, 1> *lvlSizesRef,
  StridedMemRefType<int8_t, 1> *lvlTypesRef,
  StridedMemRefType<index_type, 1> *lvl2dimRef,
  StridedMemRefType<index_type, 1> *dim2lvlRef, int ptrTp,
  int indTp, int valTp, int action, int8_t* ptr) {
    return (int8_t*) _mlir_ciface_newSparseTensor(dimSizesRef, lvlSizesRef,
      reinterpret_cast<StridedMemRefType<DimLevelType, 1>*>(lvlTypesRef),
      lvl2dimRef, dim2lvlRef, (OverheadType) ptrTp, (OverheadType) indTp,
      (PrimaryType) valTp, (Action) action, ptr);
  }
#endif

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
      size_t extents[8] = {0};
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
            Kokkos::abort("Cannot convert non-contiguous StridedMemRefType to LayoutLeft Kokkos::View");
          expectedStride *= smr.sizes[i];
        }
      }
      else if constexpr(std::is_same_v<Layout, Kokkos::LayoutRight>)
      {
        int64_t expectedStride = 1;
        for(int i = int(V::rank) - 1; i >= 0; i--)
        {
          if(expectedStride != smr.strides[i])
            Kokkos::abort("Cannot convert non-contiguous StridedMemRefType to LayoutRight Kokkos::View");
          expectedStride *= smr.sizes[i];
        }
      }
      return V(&smr.data[smr.offset], layout);
    }

  struct DualViewBase
  {
    virtual ~DualViewBase() {}
    virtual void syncHost() = 0;
    virtual void syncDevice() = 0;
    bool modified_host = false;
    bool modified_device = false;
  };

  template<typename DataType, typename Layout>
    struct DualView : public DualViewBase
  {
    using HostView = Kokkos::View<DataType, Layout, Kokkos::DefaultHostExecutionSpace>;
    using DeviceView = Kokkos::View<DataType, Layout, Kokkos::DefaultExecutionSpace>;

    static constexpr bool deviceAccessesHost = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;
    static constexpr bool hostAccessesDevice = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;

    // Default constructor makes empty views and self as parent.
    DualView() : device_view(), host_view(), parent(this) {}

    // Constructor for allocating a new view.
    // Does not actually allocate anything yet; instead 
    DualView(
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
      parent = this;
    }

    // Constructor which is given explicit device and host views, and a parent.
    // This can be used for subviewing/casting operations.
    // Note: d,h should view the same memory as parent, but they can
    // have a different data type and layout.
    DualView(DeviceView d, HostView h, DualViewBase* parent_)
      : device_view(d), host_view(h), parent(parent_)
    {}

    // Constructor for a host view from an external source (e.g. python)
    DualView(HostView h)
    {
      modified_host = true;
      if constexpr(deviceAccessesHost) {
        device_view = DeviceView(h.data(), h.layout());
      }
      else {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, h.label() + "_dev"), h.layout());
      }
      host_view = h;
      parent = this;
    }

    // Copy-assignment equivalent to the above constructor.
    // Shallow copying a temporary DualView to a persistent one leaves the
    // persistent one in an invalid state, since its parent pointer still points to the temporary.
    //
    // Shallow-copy from one persistent DualView to another persistent or temporary is OK, as long
    // as the lifetime of original covers the lifetime of the copy.
    DualView& operator=(const HostView& h)
    {
      modified_host = true;
      if constexpr(deviceAccessesHost) {
        device_view = DeviceView(h.data(), h.layout());
      }
      else {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, h.label() + "_dev"), h.layout());
      }
      host_view = h;
      parent = this;
      return *this;
    }

    void modifyHost()
    {
      parent->modified_host = true;
    }

    void modifyDevice()
    {
      parent->modified_device = true;
    }

    void syncHost() override
    {
      if (device_view.data() == host_view.data()) {
        if(parent->modified_device) {
          parent->modified_device = false;
          Kokkos::fence();
        }
      }
      else if (parent->modified_device) {
        if(parent == this) {
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
        if(parent == this) {
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

    void deallocate() {
      device_view = DeviceView();
      host_view = HostView();
    }

    DeviceView device_view;
    HostView host_view;
    DualViewBase* parent;
  };
} // namespace LAPIS

