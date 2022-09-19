from docutils import nodes, utils


def setup(app):
    def repo_link_role(
        name, rawtext, text, lineno, inliner, options=None, content=None
    ):
        if options is None:
            options = {}
        node = nodes.reference(
            rawtext,
            utils.unescape(text),
            refuri=app.config.repo_link,
            **options
        )
        return [node], []

    app.add_config_value('repo_link', '', False)
    app.add_role('repo-link', repo_link_role)
