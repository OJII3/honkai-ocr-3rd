{
  description = "崩壊3rdの実況動画からダイアログを文字起こしする開発環境";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem = { pkgs, ... }:
        let
          tessdata = pkgs.runCommand "honkai-ocr-tessdata" {} ''
            mkdir -p $out/share/tessdata
            ln -s ${pkgs.tesseract.languages.jpn} $out/share/tessdata/jpn.traineddata
          '';
        in {
          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.ffmpeg
              pkgs.python311
              pkgs.stdenv.cc.cc
              pkgs.tesseract
              pkgs.uv
              pkgs.yt-dlp
              tessdata
            ];

            TESSDATA_PREFIX = "${tessdata}/share/tessdata";
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib pkgs.glib pkgs.libGL ];

            shellHook = ''
              unset PYTHONPATH
              if [ ! -d .venv ]; then
                uv venv --python ${pkgs.python311}/bin/python
              fi
              uv sync --dev
            '';
          };
        };
    };
}
