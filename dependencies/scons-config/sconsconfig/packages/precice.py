import glob
import multiprocessing
import os
import shlex

from .Package import Package


class precice(Package):

  def __init__(self, **kwargs):
    defaults = {
        'download_url': 'https://github.com/precice/precice/archive/refs/tags/v3.3.0.zip',
    }
    defaults.update(kwargs)
    super(precice, self).__init__(**defaults)
    self.ext = '.cpp'
    self.set_rpath = True
    self.check_text = r'''
    #include <iostream>
    #include <cstdlib>
    #include <fstream>
    #include <precice/precice.hpp>

    int main()
    {
      std::ofstream file("install_precice-config.xml");
      file << R"(<?xml version="1.0"?>
<precice-configuration>

    <data:scalar name="Data"/>
    <mesh name="Mesh1" dimensions="3">
      <use-data name="Data"/>
    </mesh>
    <mesh name="Mesh2" dimensions="3">
      <use-data name="Data"/>
    </mesh>

    <participant name="Participant1">
      <provide-mesh name="Mesh1"/>
      <write-data name="Data" mesh="Mesh1"/>
    </participant>

    <participant name="Participant2">
      <receive-mesh name="Mesh1" from="Participant1"/>
      <provide-mesh name="Mesh2"/>
      <read-data name="Data" mesh="Mesh2"/>
      <mapping:nearest-neighbor
        direction="read"
        from="Mesh1"
        to="Mesh2"
        constraint="consistent" />
    </participant>

    <m2n:sockets acceptor="Participant1" connector="Participant2" network="lo" />
    <coupling-scheme:serial-explicit>
      <participants first="Participant1" second="Participant2"/>
      <time-window-size value="0.01"/>
      <max-time value="0.05"/>
      <exchange data="Data" mesh="Mesh1" from="Participant1" to="Participant2"/>
    </coupling-scheme:serial-explicit>

</precice-configuration>
)";
      file.close();

      precice::Participant participant("Participant1","install_precice-config.xml",0,1);
      //participant.initialize();
      //participant.finalize();

      int ret = system("rm -f install_precice-config.xml");
      (void)ret;
      return EXIT_SUCCESS;
    }
'''

    self.number_output_lines = int(15536 * 1.5)
    self.libs = ['precice']
    self.extra_libs = [
        [],
        ['boost_filesystem', 'boost_log_setup', 'boost_log',
         'boost_program_options', 'boost_system', 'boost_thread',
         'boost_unit_test_framework', 'dl', 'boost_regex'],
        ['boost_atomic', 'boost_chrono', 'boost_filesystem',
         'boost_log_setup', 'boost_log', 'boost_prg_exec_monitor',
         'boost_program_options', 'boost_system', 'boost_test_exec_monitor',
         'boost_thread', 'boost_unit_test_framework', 'dl', 'boost_regex'],
    ]
    self.headers = ['precice/precice.hpp']

  @staticmethod
  def _prefix_candidates():
    """Return likely package prefixes from the currently loaded environment."""
    candidates = []
    variables = (
        'CMAKE_PREFIX_PATH', 'CPATH', 'CPLUS_INCLUDE_PATH',
        'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'PATH',
    )
    for variable in variables:
      for entry in os.environ.get(variable, '').split(os.pathsep):
        if not entry:
          continue
        path = os.path.realpath(entry)
        # Environment variables often contain <prefix>/bin, <prefix>/lib,
        # <prefix>/include, or <prefix>/include/eigen3. Check their ancestors.
        for _ in range(4):
          if path and path != os.path.dirname(path):
            candidates.append(path)
            path = os.path.dirname(path)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))

  @classmethod
  def _find_package_prefix(cls, package, env_names, config_patterns,
                           required_patterns=()):
    """Find a CMake package prefix without relying on fragile module macros."""
    candidates = []
    for name in env_names:
      value = os.environ.get(name)
      if value:
        value = os.path.realpath(value)
        # *_DIR normally names the CMake config directory, not the prefix.
        if name.lower().endswith('_dir'):
          for _ in range(3):
            value = os.path.dirname(value)
        candidates.append(value)
    candidates.extend(cls._prefix_candidates())

    # Last-resort discovery for Spack installations on clusters. The patterns
    # are deliberately constrained to package/version directories.
    spack_roots = glob.glob(
        '/cluster/software/SPACK/*/spack/opt/spack/linux-*/gcc-*/{}-*'.format(package)
    )
    candidates.extend(sorted(spack_roots, reverse=True))

    for prefix in dict.fromkeys(candidates):
      if not os.path.isdir(prefix):
        continue
      configs = []
      for pattern in config_patterns:
        configs.extend(glob.glob(os.path.join(prefix, pattern)))
      if not configs:
        continue
      if any(not glob.glob(os.path.join(prefix, pattern))
             for pattern in required_patterns):
        continue
      return os.path.realpath(prefix), os.path.realpath(os.path.dirname(configs[0]))

    raise RuntimeError(
        'Could not locate a usable {} installation. Load its module or set one '
        'of: {}.'.format(package, ', '.join(env_names))
    )

  @staticmethod
  def _require(path, description):
    if not os.path.exists(path):
      raise RuntimeError('{} not found: {}'.format(description, path))
    return os.path.realpath(path)

  def check(self, ctx):
    env = ctx.env
    ctx.Message('Checking for preCICE ...       ')
    self.check_options(env)

    opendihu_home = os.path.realpath(
        os.environ.get('OPENDIHU_HOME', os.path.join(os.getcwd(), '..'))
    )
    dependencies = os.path.join(opendihu_home, 'dependencies')

    petsc_prefix = self._require(
        os.path.join(dependencies, 'petsc', 'install'), 'PETSc prefix')
    petsc_pc_dir = self._require(
        os.path.join(petsc_prefix, 'lib', 'pkgconfig'),
        'PETSc pkg-config directory')

    libxml2_prefix = self._require(
        os.path.join(dependencies, 'libxml2', 'install'), 'libxml2 prefix')
    libxml2_pc_dir = self._require(
        os.path.join(libxml2_prefix, 'lib', 'pkgconfig'),
        'libxml2 pkg-config directory')
    libxml2_include = self._require(
        os.path.join(libxml2_prefix, 'include', 'libxml2'),
        'libxml2 include directory')
    libxml2_library = self._require(
        os.path.join(libxml2_prefix, 'lib', 'libxml2.so'), 'libxml2 library')

    boost_prefix, boost_dir = self._find_package_prefix(
        'boost-1.82.0',
        ('BOOST_ROOT', 'BOOST_PREFIX', 'Boost_ROOT', 'Boost_DIR'),
        ('lib/cmake/Boost-*/BoostConfig.cmake',),
        # Reject the cluster build containing only serialization/system.
        ('lib/cmake/boost_log-*/boost_log-config.cmake',
         'lib/cmake/boost_filesystem-*/boost_filesystem-config.cmake',
         'lib/cmake/boost_program_options-*/boost_program_options-config.cmake'),
    )

    eigen_prefix, eigen_dir = self._find_package_prefix(
        'eigen-3.4.0',
        ('EIGEN3_ROOT', 'EIGEN3_PREFIX', 'Eigen3_ROOT', 'Eigen3_DIR'),
        ('share/eigen3/cmake/Eigen3Config.cmake',
         'lib/cmake/eigen3/Eigen3Config.cmake'),
    )

    old_pkg_config = os.environ.get('PKG_CONFIG_PATH', '')
    pkg_config_path = os.pathsep.join(
        p for p in (libxml2_pc_dir, petsc_pc_dir, old_pkg_config) if p)

    old_ld_library = os.environ.get('LD_LIBRARY_PATH', '')
    ld_library_path = os.pathsep.join(
        p for p in (os.path.join(boost_prefix, 'lib'),
                    os.path.join(libxml2_prefix, 'lib'),
                    os.path.join(petsc_prefix, 'lib'),
                    old_ld_library) if p)

    cmake_prefix_path = ';'.join(
        (boost_prefix, eigen_prefix, libxml2_prefix, petsc_prefix))
    jobs = max(1, int(env.get('PRECICE_JOBS', min(8, multiprocessing.cpu_count()))))

    q = shlex.quote
    cmake = ctx.env['cmake']
    configure = (
        'env PKG_CONFIG_PATH={pkg} LD_LIBRARY_PATH={ld} '
        'PETSC_DIR={petsc} PETSC_ARCH= '
        '{cmake} -S ${{SOURCE_DIR}} -B ${{SOURCE_DIR}}/build '
        '-DCMAKE_INSTALL_PREFIX=${{PREFIX}} '
        '-DCMAKE_BUILD_TYPE=Release '
        '-DPRECICE_RELEASE_WITH_DEBUG_LOG=ON '
        '-DBUILD_TESTING=OFF '
        '-DPRECICE_FEATURE_PYTHON_ACTIONS=OFF '
        '-DBoost_ROOT={boost_root} '
        '-DBoost_DIR={boost_dir} '
        '-DEigen3_DIR={eigen_dir} '
        '-DPETSC_DIR={petsc} '
        '-DLIBXML2_INCLUDE_DIR={xml_include} '
        '-DLIBXML2_INCLUDE_DIRS={xml_include} '
        '-DLIBXML2_LIBRARY={xml_library} '
        '-DCMAKE_PREFIX_PATH={prefixes}'
    ).format(
        pkg=q(pkg_config_path), ld=q(ld_library_path), petsc=q(petsc_prefix),
        cmake=cmake, boost_root=q(boost_prefix), boost_dir=q(boost_dir),
        eigen_dir=q(eigen_dir), xml_include=q(libxml2_include),
        xml_library=q(libxml2_library), prefixes=q(cmake_prefix_path),
    )

    self.set_build_handler([
        'mkdir -p ${PREFIX}/include ${SOURCE_DIR}/build',

        # Some OpenDiHu libxml2 installations place headers directly in
        # include/libxml2. Make <libxml/parser.h> work without copying files.
        'if [ -d {inc} ] && [ ! -e {inc}/libxml ]; then ln -s . {inc}/libxml; fi'.format(
            inc=q(libxml2_include)),

        # A previous failed configure can retain paths to unrelated Spack
        # packages. Always discard only preCICE's generated CMake cache.
        'rm -f ${SOURCE_DIR}/build/CMakeCache.txt && '
        'rm -rf ${SOURCE_DIR}/build/CMakeFiles ${SOURCE_DIR}/build/validation',

        configure,
        '{cmake} --build ${{SOURCE_DIR}}/build --parallel {jobs} '
        '--target precice install'.format(cmake=cmake, jobs=jobs),
    ])

    # Exactly one check/build attempt. The original file called this twice
    # unconditionally and then again in its fallback, causing repeated builds.
    res = super(precice, self).check(ctx)

    if not res[0]:
      ctx.Log(
          '\n\nInstallation of preCICE failed. Rebuild with\n'
          '  make clean; scons PRECICE_REBUILD=True\n\n'
          'Resolved dependency prefixes were:\n'
          '  Boost: {}\n  Eigen3: {}\n  PETSc: {}\n  libxml2: {}\n\n'.format(
              boost_prefix, eigen_prefix, petsc_prefix, libxml2_prefix))

    self.check_required(res[0], ctx)
    ctx.Result(res[0])
    return res[0]
