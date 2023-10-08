using FastLevenbergMarquardt
using Documenter

DocMeta.setdocmeta!(FastLevenbergMarquardt, :DocTestSetup, :(using FastLevenbergMarquardt); recursive=true)

makedocs(;
    modules=[FastLevenbergMarquardt],
    authors="kamesy <ckames@physics.ubc.ca> and contributors",
    repo="https://github.com/kamesy/FastLevenbergMarquardt.jl/blob/{commit}{path}#{line}",
    sitename="FastLevenbergMarquardt.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kamesy.github.io/FastLevenbergMarquardt.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kamesy/FastLevenbergMarquardt.jl",
    devbranch="main",
)
