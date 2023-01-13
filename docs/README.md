# Kernl static documentation site

## Version management of Material for MkDocs for Insiders 

Updates to the Material for MkDocs for Insiders docker image are automatic.

To avoid any breaking changes, we target a fixed version.
Please check the version of the image in the [private registry](https://github.com/orgs/ELS-RD/packages/container/package/mkdocs-material-insiders), 
and check the [changelog](https://squidfunk.github.io/mkdocs-material/insiders/changelog/). 

If necessary, perform the version upgrade or the [upgrade](https://squidfunk.github.io/mkdocs-material/upgrade/) the project.

## Preview the static site locally 

The easiest and least intrusive way is to use docker.

```shell
# Building a mkdocs image with the mkdocstrings plugin
docker build -t mkdocstrings -f docs/Dockerfile.mkdocstrings docs

# Previewing the site in watch mode
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs mkdocstrings

# Build the static site
docker run --rm -it -v ${PWD}:/docs mkdocstrings build
```

## Notes for developers

### Use variabilized classes taking into account the dark and light modes

To use the variabilized classes taking into account the dark and light modes (example: `color: var(--md-default-fg-color)`).
You need to add a `data-md-color-scheme="slate" | "default"` metadata to the enclosing elements
