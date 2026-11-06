# Optional detector DLL. Not linked into mpv/libmpv; loaded only after explicit
# addon activation. The legacy analysis core is GPL-2.0-or-later.
add_languages('cpp', required: true)
mv_cpp = meson.get_compiler('cpp')
mv_cpp_args = ['-DAJN_MVTOOLS_BUILD', '-D_CRT_SECURE_NO_WARNINGS']
mv_kernel_args = mv_cpp.get_id() == 'msvc' ? ['/FIcstddef'] : ['-include', 'cstddef']
if host_machine.cpu_family() == 'x86_64'
    mv_kernel_args += ['-DAJN_MVTOOLS_SSE2']
endif
mv_link_args = mv_cpp.get_id() == 'gcc' ? ['-static-libstdc++', '-static-libgcc'] : []
mvtools_library = shared_library('ajn-mvtools',
    files('detector.cpp', 'vendor/CopyCode.cpp', 'vendor/Fakery.cpp',
          'vendor/GroupOfPlanes.cpp', 'vendor/Luma.cpp', 'vendor/MVFrame.cpp',
          'vendor/PlaneOfBlocks.cpp', 'vendor/SADFunctions.cpp'),
    name_prefix: '', cpp_args: mv_cpp_args + mv_kernel_args, link_args: mv_link_args,
    install: true, install_dir: get_option('bindir'))
install_data('LICENSE', 'UPSTREAM.md', install_dir: join_paths(get_option('datadir'), 'licenses', 'ajn-mvtools'))
if get_option('tests')
    mvtools_test = executable('ajn-mvtools-test',
        files('test_detector.cpp', 'vendor/SADFunctions.cpp', 'vendor/Luma.cpp'),
        cpp_args: mv_kernel_args, link_with: mvtools_library)
    test('ajn-mvtools-detector', mvtools_test)
endif
