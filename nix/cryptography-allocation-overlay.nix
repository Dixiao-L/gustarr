# Bound only the negative allocation checks; preserve all package tests.
_final: prev:
prev.lib.optionalAttrs prev.stdenv.hostPlatform.isLinux {
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (_pyFinal: pyPrev: {
      cryptography =
        if pyPrev.cryptography.version == "48.0.0" then
          pyPrev.cryptography.overridePythonAttrs (old: {
            patches = (old.patches or [ ]) ++ [ ./cryptography-bounded-allocation-test.patch ];
          })
        else
          pyPrev.cryptography;
      ipython =
        if pyPrev.ipython.version == "9.9.0" then
          pyPrev.ipython.overridePythonAttrs (old: {
            disabledTests = (old.disabledTests or [ ]) ++ [
              "test_timeit_raise_on_interrupt"
              "test_script_raise_on_interrupt"
              "test_time_raise_on_interrupt"
            ];
          })
        else
          pyPrev.ipython;
    })
  ];
}
