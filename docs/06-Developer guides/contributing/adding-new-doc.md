---
id: adding-new-doc
title: Adding a new docs page
---

All Towhee documentation is written via [markdown](https://www.markdownguide.org/) pages. If you're looking to update a page, please visit the [main Towhee repo](https://github.com/towhee-io/towhee/tree/main/docs) and find the corresponding `.md` file. The sidebar structure on the [docs page](https://docs.towhee.io) matches the directory structure in the `docs` subfolder of the main Towhee repo, so if you're having trouble finding the right file, try going through subdirectories as if they were menu items on the sidebar.

### Adding a new documentation page

Adding a new documentation page is also simple - simply find the right menu item you'd like to put it under and add a new file. The header of your new file should contain two fields: and `id`, which is a unique identifier for your document, and `title`, which will appear as an `<h1>` on the first line of your page. Note that the `id` field should correspond to the name of your file. For example, if your filename is `a-unique-page-id.md`, then the header would look something like this:

```md
---
id: a-unique-page-id
title: Title of your document
---
```

Once that's done, you can add the remaining content as a regular markdown file. If you've implemented a new pipeline type or other large feature, please test your changes locally first before submitting a pull request to the main Towhee repo. You can do so with any markdown viewer.
