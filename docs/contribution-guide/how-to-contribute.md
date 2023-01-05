# Contributing guidelines

Our goal is to **make kernl development smooth**, **stable** and **open**.
We can't do this without our user community!
Please read this document before contributing!

## Code of Conduct

Kernl has adopted the **Contributor Covenant** as its **[Code of Conduct](code-of-conduct.md)**, and we expect project participants to adhere to it. 
Please read the **[full text](code-of-conduct.md)** so that you can understand what actions will and will not be tolerated.

## Open development

Kernl uses **[GitHub](https://github.com/ELS-RD/kernl)** as a source of truth. The core team works directly on it. All changes are public.

## Writing and formatting text

In a global way, texts in Issues, PRs and commits should be written **in English**, **with care**, **consistency**, and **clarity**.
Use (**[Markdown](https://www.markdownguide.org/basic-syntax/)**) to structure your texts, and to make them more readable.

## Development Process

### Ask a question / share your ideas or new features

If you have a question, an idea for a new feature, or an optimization, you can **use the dedicated [discussion forum](todo)**.

??? example "@reviewer: proposal to be validated"
    ```
    @team:  what about creating a dedicated discussion forum?
            like Remix: https://github.com/remix-run/remix/discussions or Vite for example
                Proposed categories:
                - General (That's what we communicate)
                - Ideas (open to ideas in general)
                - Q&A
                - Feature discussion (dedicated to features)
            @todo:  If you agree, create the discussion forum and add the link above.
    ```

### Issues

#### Bugs

If you think you've found a bug but don't have the time to send a PR, please:

- First, **make sure the bug has not already been reported** in the **[issue tracker](https://github.com/ELS-RD/kernl/issues)**.
- If you don't find a similar issue, **[open a new one](todo)**.

**Bug reports without a reproduction will be immediately closed.**

!!! tip "tips for writing your Issue"

    - [x] Give a concise and explicit title to the problem so that you can quickly understand what it is about.
    - [x] Describe the problem clearly and in detail. Explain what is not working as expected, what you expected to happen and what happened instead. Include as much information as possible, all the steps you need to take to reproduce the bug, etc. If you have ideas on how to fix the problem, feel free to include them in your description.

!!! example "@reviewer: see bug[report template](https://github.com/ELS-RD/kernl/blob/feat/contribution-guide-doc/.github/ISSUE_TEMPLATE/bug.yml)"

#### Proposals

If you intend to **make non-trivial changes to existing implementations**, we recommend that you **file an Issue** with the **[proposal template](todo)**. This allows us to reach agreement on your proposal before you put any effort into it.

!!! example "@reviewer: see [proposal template](https://github.com/ELS-RD/kernl/blob/feat/contribution-guide-doc/.github/ISSUE_TEMPLATE/proposal.yml)"

#### Feature requests

If you would like to request a **new feature** or **an enhancement**, 
but are not planning to open a Pull Request, you can **[open a dedicated issue](todo)**.
You can also use the **[discussion forum](todo)** for feature requests that you would like to discuss.

!!! example "@reviewer: see [feature request template](https://github.com/ELS-RD/kernl/blob/feat/contribution-guide-doc/.github/ISSUE_TEMPLATE/feature.yml)"

## Development

### Branch Organization

Kernl works with a `main` branch. This is the source of truth.

Each **new Pull Request**, for a new feature, bug fix, or other, must be done on **a new branch**.

#### Template and example of a new branch name

The name of a new branch must respect the following syntax:
```
{prefix}/{name-identifying-the-object-of-the-pull-request}
``` 

??? question "Which prefixes and characters are allowed?"

    Prefixes should respect, as much as possible, the values of **[Conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)**
    For example:

    - `feat`: new functionality, optimization or implementation of a proposal.
    - `fix`: bug fix.
    - `docs`: ajout ou modification de la documentation.
    - `refactor`: a change in the code that does not result in any difference in behavior.
    - `test`: adding tests, refactoring tests. No production code change.
    - `chore`: upgrading dependencies, releasing new versions. Tasks that are regularly done for maintenance purposes.
    - `misc`: anything else that doesn't change production code, yet is not test or chore. e.g. updating GitHub actions workflow.

    Allowed characters:
        ```regexp
        [a-z_-]
        ```
    !!! example "Examples"
        - feat/debugger
        - feat/backward_layernorm
        - refactor/refactor-kernels
        - docs/contribution-guide-doc

### Installation?

### Code convention?

### Code quality?

### Testing?

### Documentation?

## Pull Requests

### Writing a good commit message

Commit messages must respect the **[Conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)** specification.





