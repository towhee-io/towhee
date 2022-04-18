---
id: contributing-guide
title: Adding a new docs page
---

If you've implemented a new pipeline time, or other large feature, please test your changes locally first before submitting a pull request to the main Towhee repo:

1. Make sure you have `node` and `yarn` installed. On Debian-based Linux distros, you can use the built-in package manager `apt`:

```shell
$ sudo apt install nodejs
$ npm install --global yarn
```

If you're on MacOS, you can use [`brew`](https://brew.sh/):

```shell
% brew install node
% brew install yarn
```

2. Fork the [`towhee-docs`](https://github.com/towhee-io/towhee-docs) repo. Once that's done, clone your forked version of `towhee-docs`:

```shell
$ git clone https://github.com/<your_username>/towhee-docs.git
$ cd towhee-docs
```

3. There's a `towhee` subfolder within `towhee-docs` that currently points to the main Towhee repo. Delete that folder and create a symlink to your development directory. For example:

```shell
$ rm -r towhee
$ ln -s /home/<your_username>/towhee towhee
```

4. With the above three steps complete, you can now view changes to the local documentation with:

```shell
$ yarn
$ yarn start
```

5. The header for newly added pages should look something like this:

```md
---
id: page-id
title: This is a dummy title
---
```

**IMPORTANT**: Be sure to also update the `sidebar.js` file within the `towhee/docs` directory with your newly added page. If you don't do this, your new page won't show up on the Towhee docs page.

6. Before submitting a pull request with your changes, please repeat step 4 and test your changes locally.
