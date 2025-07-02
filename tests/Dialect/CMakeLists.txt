enable_testing()

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_test(NAME Kokkos_Dialect COMMAND lit -v Kokkos)
add_test(NAME Transformations COMMAND lit -v Transforms)
#add_test(NAME PartTensor_Dialect COMMAND lit -v PartTensor)
