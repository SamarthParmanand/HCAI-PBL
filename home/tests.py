"""Site-wide tests: the shell every app shares.

These guard the things that are easy to break silently when a template changes:
the UI library actually being present and local, every page using the same shell,
and no page pointing at a static file that is not there.
"""

import os
import re

from django.conf import settings
from django.test import TestCase
from django.urls import reverse

from .views import PROJECTS, STUDENTS

# Every page that renders without needing an uploaded dataset in the session.
PUBLIC_PAGES = [
    "home:index",
    "project1:index",
    "project2:index",
    "demos:upload",
    "demos:plot",
]


class HomePageTests(TestCase):

    def test_the_group_is_listed(self):
        page = self.client.get(reverse("home:index"))
        for student in STUDENTS:
            self.assertContains(page, student["name"])
            self.assertContains(page, student["matriculation"])

    def test_every_project_is_linked(self):
        page = self.client.get(reverse("home:index"))
        for project in PROJECTS:
            self.assertContains(page, reverse(project["url_name"]))
            self.assertContains(page, project["name"])

    def test_the_group_list_is_shared_with_the_report(self):
        """The Project 2 report cites the same list, so it cannot drift."""
        from project2 import report
        self.assertIs(report.STUDENTS, STUDENTS)


class RootUrlTests(TestCase):
    """`HCAI-project_01.pdf` 1.1 tells the reader to open http://127.0.0.1:8000/.

    That is the first address a marker types, and for a while it returned 404:
    every app was mounted under a prefix and nothing answered at the bare root.
    """

    def test_the_bare_root_reaches_the_hub(self):
        page = self.client.get("/", follow=True)
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.redirect_chain[-1][0], reverse("home:index"))
        for project in PROJECTS:
            self.assertContains(page, reverse(project["url_name"]))


class SiteShellTests(TestCase):
    """The UI library and the shared layout."""

    def pages(self):
        for name in PUBLIC_PAGES:
            yield name, self.client.get(reverse(name)).content.decode()

    def test_bootstrap_is_vendored_not_fetched_from_a_cdn(self):
        """The submission has to render with no network connection."""
        for name, page in self.pages():
            self.assertIn("/static/vendor/bootstrap/bootstrap.min.css", page, name)
            self.assertNotIn("cdn.jsdelivr.net", page, name)
            self.assertNotIn("cdnjs.cloudflare.com", page, name)
            self.assertNotIn("//stackpath", page, name)

    def test_the_vendored_files_exist_on_disk(self):
        for relative in ("vendor/bootstrap/bootstrap.min.css",
                         "vendor/bootstrap/bootstrap.bundle.min.js",
                         "style.css"):
            path = os.path.join(settings.BASE_DIR, "static", *relative.split("/"))
            self.assertTrue(os.path.exists(path), relative)
            self.assertGreater(os.path.getsize(path), 100, relative)

    def test_every_static_reference_resolves(self):
        """A typo in a static path fails silently in the browser; not here."""
        for name, page in self.pages():
            for href in re.findall(r'(?:href|src)="(/static/[^"]+)"', page):
                relative = href.split("/static/", 1)[1]
                path = os.path.join(settings.BASE_DIR, "static", *relative.split("/"))
                self.assertTrue(os.path.exists(path), f"{name}: {href} is missing")

    def test_every_page_uses_the_shared_shell(self):
        for name, page in self.pages():
            self.assertIn('<nav class="navbar', page, name)
            self.assertIn('class="container-xl', page, name)
            self.assertIn("</html>", page, name)

    def test_the_footer_sits_at_the_bottom_and_is_centred(self):
        """A short page used to leave the footer floating mid-viewport.

        The sticky-footer pattern needs all three parts: a full-height flex
        column on the body, a main that grows into the spare room, and mt-auto
        on the footer. Any one of them missing and the footer drifts back up,
        which is invisible on a long page and obvious on a short one.
        """
        for name in PUBLIC_PAGES:
            page = self.client.get(reverse(name)).content.decode()
            self.assertIn("d-flex flex-column min-vh-100", page, name)
            self.assertIn("flex-grow-1", page, name)
            footer = page[page.index("<footer"):page.index("</footer>")]
            self.assertIn("mt-auto", footer, name)
            self.assertIn("text-center", footer, name)

    def test_every_page_is_responsive(self):
        for name, page in self.pages():
            self.assertIn('name="viewport"', page, name)

    def test_no_template_syntax_leaks_on_any_page(self):
        for name, page in self.pages():
            for leaked in ("{#", "#}", "{%", "%}", "{{", "}}"):
                self.assertNotIn(leaked, page, f"{leaked} leaked into {name}")

    def test_no_page_still_references_the_removed_stylesheets(self):
        """The bespoke per-app sheets were replaced by Bootstrap plus one theme."""
        for name, page in self.pages():
            self.assertNotIn("project1/style.css", page, name)
            self.assertNotIn("project2/style.css", page, name)

    def test_selects_contain_only_options(self):
        """A block element inside a <select> renders as unusable markup."""
        for name, page in self.pages():
            for block in page.split("<select")[1:]:
                inner = block.split("</select>")[0]
                for forbidden in ("<p", "<div", "<form"):
                    self.assertNotIn(forbidden, inner, name)

    def test_every_page_titles_itself(self):
        for name, page in self.pages():
            title = re.search(r"<title>(.*?)</title>", page, re.S)
            self.assertIsNotNone(title, name)
            self.assertTrue(title.group(1).strip(), name)


class ThemeTests(TestCase):
    """The theme layer only re-points Bootstrap; it must not fight it."""

    def stylesheet(self):
        path = os.path.join(settings.BASE_DIR, "static", "style.css")
        return open(path, encoding="utf-8").read()

    def test_vendor_prefixed_rules_are_not_grouped(self):
        """A browser drops a whole comma-group if one selector is unknown to it.

        Grouping `::-moz-range-thumb` with an ordinary selector therefore silently
        kills the ordinary one in Chrome, which is how the themed checkbox was lost
        the first time.
        """
        for rule in re.findall(r"([^{}]+)\{", self.stylesheet()):
            selectors = [part.strip() for part in rule.split(",")]
            if len(selectors) < 2:
                continue
            prefixed = [s for s in selectors if "::-moz-" in s or "::-webkit-" in s]
            self.assertFalse(
                prefixed and len(selectors) > len(prefixed),
                f"vendor-prefixed selector grouped with others: {rule.strip()}")

    def test_the_theme_matches_the_chart_palette(self):
        """A figure and the page around it must agree on the greys."""
        from project1 import plots
        sheet = self.stylesheet()
        self.assertIn(plots.INK.lower(), sheet.lower())
        self.assertIn(plots.MUTED.lower(), sheet.lower())
        self.assertIn(plots.GRID.lower(), sheet.lower())
