"""Access control for retrieval.

The rule this module exists to enforce: **a document the viewer may not read
must never enter the candidate set.** Not "must be dropped from the answer" --
never retrieved at all.

Why that distinction is the whole point:

  post-filtering  retrieve the top 20, then remove what the viewer cannot see.
                  Broken three ways. You ask for 5 results and get 2. The
                  forbidden documents still consumed ranking slots, so the
                  results you *do* get are worse. And the moment anything
                  downstream forgets to apply the filter -- a debug endpoint, a
                  log line, a cached candidate list -- the content leaks.

  pre-filtering   restrict the search space to what the viewer may read, then
                  retrieve. The forbidden content is never in memory as a
                  candidate, so there is nothing to leak and nothing to
                  remember to strip.

This is the same reasoning as row-level security in Postgres: the filter
belongs in the query, not in the application code that reads the results. In
pgvector it is a WHERE clause before the ORDER BY:

    SELECT id, content
    FROM chunks
    WHERE visibility = ANY(%(allowed)s)      -- <- pre-filter
    ORDER BY embedding <=> %(q)s
    LIMIT 20;
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Visibility levels a document can carry, loosest first.
PUBLIC = "public"      # any employee
MANAGER = "manager"    # people managers
HR = "hr"              # HR team and executives
SELF = "self"          # about one specific person; only that person (+ HR)

ALL_LEVELS = {PUBLIC, MANAGER, HR, SELF}

# What each role may read. Deliberately explicit rather than a hierarchy with
# integer ranks: "manager" outranks "employee" for team documents but must NOT
# thereby gain access to salary bands. Real permission models are lattices, not
# ladders, and encoding them as a ladder is how privilege-escalation bugs are
# born.
ROLE_GRANTS: dict[str, set[str]] = {
    "employee": {PUBLIC},
    "manager": {PUBLIC, MANAGER},
    "hr": {PUBLIC, MANAGER, HR, SELF},
    "admin": {PUBLIC, MANAGER, HR, SELF},
}


@dataclass
class Principal:
    """Who is asking. Every retrieval call needs one."""

    employee_id: str
    role: str = "employee"
    name: str = ""
    # Documents about a specific person carry `subject`; a viewer may read
    # their own even when the level is SELF.
    extra_subjects: set[str] = field(default_factory=set)

    @property
    def allowed_levels(self) -> set[str]:
        return ROLE_GRANTS.get(self.role, {PUBLIC})

    def may_read(self, visibility: str, subject: str | None = None) -> bool:
        """The single authority on whether this principal can see a chunk."""
        visibility = (visibility or PUBLIC).lower()

        # Unknown level -> deny. Fail closed: a typo in a document's
        # front-matter must not silently make it world-readable.
        if visibility not in ALL_LEVELS:
            return False

        if visibility == SELF:
            # own record, or HR/admin
            if subject and (subject == self.employee_id or subject in self.extra_subjects):
                return True
            return SELF in self.allowed_levels

        return visibility in self.allowed_levels

    def describe(self) -> str:
        return f"{self.name or self.employee_id} ({self.role})"


# Convenience principals for demos and tests.
ANONYMOUS = Principal(employee_id="anon", role="employee", name="کاربر ناشناس")
