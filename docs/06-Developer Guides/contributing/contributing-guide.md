# Contributing Guide

Submitting issues, answering questions, and improving documentation are some of the many ways you can join our growing community. The Towhee community maintains a list of [Good First Issues](https://github.com/towhee-io/towhee/contribute) - feel free to browse and see if any of these issues pique your interest. Regardless of how you contribute, please remember to be respectful towards your fellow Towheeians.

### Contributing to Towhee

**Sharing a new operator or pipeline**

Pipeline and operator contributions to our Towhee Hub are just as valued as artwork, code, and documentation contributions. If you have a new model, useful script, or an `x2vec` pipeline that you'd like to share with the Towhee community, get in touch with us on Slack!

**Submitting a new issue or feature request**

Please follow the [templates](https://github.com/towhee-io/towhee/issues/new/choose) we provide for submitting bugs, enhancements, and/or feature requests; try to avoid opening blank issues whenever possible.

**Style guide**

We generally follow the [Google Python style guide](https://google.github.io/styleguide/pyguide.html) - this applies to the main `towhee-io/towhee` repo on Github as well as code uploaded to our Towhee Hub. We have some special rules regarding line length, imports, and whitespace - please take a look at the [Towhee style guide](https://github.com/towhee-io/towhee/STYLE_GUIDE.md) for more information.

**Documentation**

If you've noticed a typo or other error somewhere in the [documentation](https://docs.towhee.io), you can push an update by editing the corresponding page in the `towhee/docs` directory, submitting a pull request (see the next section), and waiting for the changes to be merged into the repo. If you need to add a new page or make other substantial documentation changes, please refer to [this page](./adding-new-doc.md).

### Pull requests

We follow a fork-and-pull model for all contributions. Before starting, we strongly recommend looking through existing PRs so you can get a feel for things.

If you're interested in contributing to the `towhee-io/towhee` codebase, follow these steps:

1. Fork [Towhee](https://github.com/towhee-io/towhee). If you've forked Towhee already, simply fetch the latest changes from upstream.

2. Clone your forked version of Towhee.

```bash
$ git clone https://github.com/<your_username>/towhee.git
$ cd towhee
```

If you've done this step before, make sure you're on the `main` branch and sync your changes.

```bash
$ git checkout main
$ git pull origin main
```

3. Think up a suitable name for your update, bugfix, or feature. Try to avoid using branch names you've already used in the past.

```bash
$ git checkout -b my-creative-branch-name
```

4. During development, you might want to run `pylint` or one of the tests. You can do so with one of the commands below:

```bash
$ pylint --rcfile pylint.conf
$ pytest tests/unittests/<test_case>.py
```

5. If you're contributing a bugfix or docfix, squash your previous `N` commits. The interactive rebase functionality provided by git will walk you through the commit squashing process.

```bash
$ git rebase -i HEAD~N
```

P.S. Don't forget to commit your changes! We use a single-phrase, periodless format for commit messages (be sure to capitalize the first character). We also sign off all of our commits - you can read more about this [here](https://dlorenc.medium.com/should-you-sign-git-commits-f068b07e1b1f).

```bash
$ git commit -sm "My awesome commit message"
```

6. Submit your pull request on Github. Folks in the community will discuss your pull request, and maintainers might ask you for some changes. This happens very frequently (including maintainers themselves), so don't worry if it happens to you as well.

Note that Towhee uses [DCOs](https://developercertificate.org/) to sign pull requests. Please ensure that the first line of your PR is as follows: Signed-off-by: Your Name your.email@domain.com
