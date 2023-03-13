# Kernl static documentation site

## Material for MkDocs Insiders version

To avoid any breaking change, we target a specific version.

For the preview ([`Dockerfile.mkdocstrings`](Dockerfile.mkdocstrings)),
and in the Github Action
workflow ([`deploy-static-site.yml`](../.github/workflows/deploy-static-site.yml))

### How to upgrade version

To take advantage of the latest features, check
the [online changelog](https://squidfunk.github.io/mkdocs-material/insiders/changelog/).

According to it, **synchronize the fork**, **update the docker image** and **the workflow**.

If necessary, take into account the breaking change by
consulting [How to upgrade](https://squidfunk.github.io/mkdocs-material/upgrade/).

## Preview the static site locally

The easiest and least intrusive way is to use docker.

### Important❗

You **need to be logged** in to github to pull the image
(see [docker login](https://docs.docker.com/engine/reference/commandline/login/)).

### About the **`CI`'s Environment variable**

An environment variable `CI` is set to `true` when deploying continuously to Github pages. This environment variable
environment variable allows the activation of the `optimize` plugin (automatic image optimization).

If you want the same behavior when running locally, you can set this variable to `true` by adding the `-e CI=true`
option to the `docker run` command.

> _Note that other optimizations, like `external_assets_exclude`, are executed automatically, both locally and in
continuous integration._

```shell
# Building a mkdocs image with the mkdocstrings plugin
docker build -t mkdocstrings -f docs/Dockerfile.mkdocstrings docs
```

```shell
# Previewing the site in watch mode
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs mkdocstrings

# Previewing the site in watch mode with optimization plugins
docker run --rm -it -p 8000:8000 -e CI=true -v ${PWD}:/docs mkdocstrings
```

```shell
# Build the static site
docker run --rm -it -v ${PWD}:/docs mkdocstrings build

# Build the static site with optimization plugins
docker run --rm -it -e CI=true -v ${PWD}:/docs mkdocstrings build
```

## Notes for developers

### Use variabilized classes taking into account the dark and light modes

To use the variabilized classes taking into account the dark and light modes (
example: `color: var(--md-default-fg-color)`).
You need to add a `data-md-color-scheme="slate" | "default"` metadata to the enclosing elements
