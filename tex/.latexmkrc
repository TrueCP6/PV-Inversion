# Keep build products out of tex/. These paths mirror the IDE run configuration in
# .idea/runConfigurations/main__LaTeX_.xml (output-path and auxil-path both point at
# tex_out), so `latexmk` run by hand from this directory behaves the same way it does
# from the IDE. They are relative to this directory, which latexmk uses as the cwd.
$out_dir = '../tex_out';
$aux_dir = '../tex_out';

# minted shells out to pygmentize; \usepackage[outputdir=../tex_out]{minted} in main.tex
# tells it where to put the cache, but it still needs the escape enabled to run at all.
$pdflatex = 'pdflatex -shell-escape %O %S';

$pdf_mode = 1;
