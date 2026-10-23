# -*- coding: utf-8 -*-
"""
The machines a run can go to, and the preferences a run uses on one of them.

Preferences keep a list of machines (``Preferences.machines``, plain dicts so
the JSON file stays readable) and the name of the default one
(``Preferences.default_machine``). Two kinds today:

* ``local``: this PC, with the Abaqus command of the Preferences. There is
  always exactly one, named LOCAL_NAME; it is not stored in the list.
* ``queue``: a PC reached through a queue folder and the remote agent
  (gui.core.remote_exec): its queue folder and its abaqus.bat.

A study picks a machine by name. `prefs_for_machine` returns a copy of the
preferences set for that machine (execution_mode, remote_queue_dir,
remote_abaqus_cmd), so the run loops, which read those fields, need no other
change. The copy's ``default_machine`` is the machine it is set for
(`machine_name`).

Preferences written before the list existed have none (None, while an
empty list means no queue machine): `machines` builds it
from their execution fields, so the queue folder already set becomes a
machine and the mode chosen becomes the default (`migrate`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

LOCAL = "local"
QUEUE = "queue"
KINDS = (LOCAL, QUEUE)
LOCAL_NAME = "This PC"
# Name given to the queue folder of preferences older than the list.
MIGRATED_NAME = "Compute PC"


@dataclass
class Machine:
    name: str
    kind: str = QUEUE
    queue_dir: str = ""
    abaqus_cmd: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "queue_dir": self.queue_dir, "abaqus_cmd": self.abaqus_cmd}

    @classmethod
    def from_dict(cls, d) -> Optional["Machine"]:
        """None for an entry that is not a dict or has no name."""
        if not isinstance(d, dict):
            return None
        name = str(d.get("name", "") or "").strip()
        kind = d.get("kind", QUEUE)
        if not name or kind not in KINDS:
            return None
        return cls(name=name, kind=kind,
                   queue_dir=str(d.get("queue_dir", "") or ""),
                   abaqus_cmd=str(d.get("abaqus_cmd", "") or ""))


def _stored(prefs) -> List[Machine]:
    out = []
    for d in getattr(prefs, "machines", None) or []:
        m = Machine.from_dict(d)
        if m is not None and m.kind == QUEUE and m.name != LOCAL_NAME \
                and all(m.name != o.name for o in out):
            out.append(m)
    return out


def machines(prefs) -> List[Machine]:
    """This PC first, then the queue machines. Preferences without a list
    get the one their execution fields describe."""
    local = Machine(LOCAL_NAME, LOCAL, "", getattr(prefs, "abaqus_cmd", ""))
    stored = _stored(prefs)
    if getattr(prefs, "machines", None) is None:
        queue = getattr(prefs, "remote_queue_dir", "") or ""
        if queue:
            stored = [Machine(MIGRATED_NAME, QUEUE, queue,
                              getattr(prefs, "remote_abaqus_cmd", "") or "")]
    return [local] + stored


def names(prefs) -> List[str]:
    return [m.name for m in machines(prefs)]


def find(prefs, name) -> Optional[Machine]:
    for m in machines(prefs):
        if m.name == name:
            return m
    return None


def default_name(prefs) -> str:
    """The default machine: the one named in the preferences when it
    exists, else the one their execution mode points to."""
    all_ = machines(prefs)
    name = getattr(prefs, "default_machine", "") or ""
    if any(m.name == name for m in all_):
        return name
    if getattr(prefs, "execution_mode", "local") == "remote" and len(all_) > 1:
        return all_[1].name
    return LOCAL_NAME


def migrate(prefs):
    """The preferences with their machine list written out (no change when
    they already have one)."""
    if getattr(prefs, "machines", None) is not None:
        return prefs
    return replace(prefs,
                   machines=[m.to_dict() for m in machines(prefs)[1:]],
                   default_machine=default_name(prefs))


def prefs_for_machine(prefs, name: str = ""):
    """A copy of `prefs` set to run on machine `name` ("" = the default
    machine). None when no machine has that name (removed from the
    preferences since it was chosen)."""
    m = find(prefs, name or default_name(prefs))
    if m is None:
        return None
    if m.kind == LOCAL:
        return replace(prefs, execution_mode="local", default_machine=m.name)
    return replace(prefs, execution_mode="remote", remote_queue_dir=m.queue_dir,
                   remote_abaqus_cmd=m.abaqus_cmd, default_machine=m.name)


def machine_name(prefs) -> str:
    """The machine a copy from `prefs_for_machine` is set for (the default
    machine for the user's own preferences)."""
    return default_name(prefs)
