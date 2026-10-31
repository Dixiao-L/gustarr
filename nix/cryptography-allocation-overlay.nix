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
    })
  ];
}
