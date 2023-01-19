# Kernl static documentation site

## Material for MkDocs Insiders version

To avoid any breaking change, we target a specific version.

For the preview (`doc/Dockerfile.mkdocstrings`),
and in the Github Action workflow (`.github/workflows/deploy-static-site.yml`)


### How to upgrade version

To take advantage of the latest features, check the [online changelog](https://squidfunk.github.io/mkdocs-material/insiders/changelog/).

According to it, **synchronize the fork**, **update the docker image** and **the project**.

If necessary, take into account the breaking change by consulting [How to upgrade](https://squidfunk.github.io/mkdocs-material/upgrade/).

## Preview the static site locally

The easiest and least intrusive way is to use docker.

```shell
# Building a mkdocs image with the mkdocstrings plugin
docker build -t mkdocstrings -f docs/Dockerfile.mkdocstrings docs
```

```shell
# Previewing the site in watch mode
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs mkdocstrings
```

```shell
# Build the static site
docker run --rm -it -v ${PWD}:/docs mkdocstrings build
```

## Notes for developers

### Use variabilized classes taking into account the dark and light modes

To use the variabilized classes taking into account the dark and light modes (example: `color: var(--md-default-fg-color)`).
You need to add a `data-md-color-scheme="slate" | "default"` metadata to the enclosing elements
