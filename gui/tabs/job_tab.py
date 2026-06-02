# -*- coding: utf-8 -*-
"""
Job tab.

For now operates in DRY-RUN mode only:
  - Generates the exact subprocess command that ABQ.run_simul would invoke,
    without launching Abaqus.
  - Displays the model_params dict in the same `repr(...)` form ABQ.py uses,
    so the user can sanity-check the serialisation before flipping on real
    execution.

A future iteration will:
  - Enable the "Run" button (currently disabled).
  - Stream Abaqus's stdout/stderr into the log panel as the job runs.
  - Optionally chain `ABQ.extract()` after a successful run.
"""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Signal, Qt, QProcess, QProcessEnvironment
from PySide6.QtGui import QFont, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QPlainTextEdit, QFormLayout,
    QSplitter, QFileDialog, QFrame, QScrollArea, QMessageBox,
)

from gui.core.model_config import ModelConfig
from gui.core.preferences import Preferences


def _section_header(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "background-color: #e8eef5; color: #1f4060; "
        "font-weight: bold; padding: 3px 6px; "
        "border-left: 3px solid #1f6fb2;"
    )
    return lbl


class JobTab(QWidget):
    """Job parameters editor + dry-run command preview.

    Emits `jobChanged()` whenever a job field is edited (so the parent
    can flip the dirty flag — even though job settings live on this tab
    instance and aren't yet persisted in the ModelConfig, we treat them
    as part of the project intent)."""

    jobChanged = Signal()

    def __init__(self, cfg: ModelConfig, prefs_getter, parent=None):
        """`prefs_getter` is a zero-arg callable returning the *current*
        Preferences. We don't store a Preferences instance here — the
        user may change it in the Preferences dialog and we want the
        next dry-run to reflect the change."""
        super().__init__(parent)
        self.cfg = cfg
        self._get_prefs = prefs_getter

        # QProcess used for Run. We create it lazily in `_run_abaqus`
        # because the user might never click Run, and a QProcess instance
        # carries a small amount of OS state we don't need until then.
        self._proc: QProcess | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ----- Upper section (job parameters + run controls) wrapped in a
        # QScrollArea so the rows stay readable when the window is short.
        # The output text edit below has its own native scroll, so we keep
        # it outside the wrapper to avoid nested scrolls. ------
        upper = QWidget()
        upper_lay = QVBoxLayout(upper)
        upper_lay.setContentsMargins(8, 8, 8, 8)
        upper_lay.setSpacing(8)
        upper_lay.addWidget(self._build_job_params_group())
        upper_lay.addWidget(self._build_run_controls())

        upper_scroll = QScrollArea()
        upper_scroll.setWidgetResizable(True)
        upper_scroll.setFrameShape(QFrame.Shape.NoFrame)
        upper_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        upper_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        upper_scroll.setWidget(upper)
        # Don't let the upper section take more than half the tab height by
        # default — leave most space for the output panel.
        upper_scroll.setMaximumHeight(220)
        outer.addWidget(upper_scroll, stretch=0)

        # ----- Output preview (native scroll inside QPlainTextEdit) -----
        outer.addWidget(self._build_output_panel(), stretch=1)

    # =====================================================================
    # Sub-groups
    # =====================================================================
    def _build_job_params_group(self) -> QGroupBox:
        g = QGroupBox("Job parameters")
        form = QFormLayout(g)

        self.fld_job_name = QLineEdit("Cutting_job")
        self.fld_job_name.setMaximumWidth(260)
        self.fld_job_name.setToolTip(
            "Used as the .inp/.odb base name. Avoid spaces and slashes;\n"
            "Abaqus is picky about job names."
        )
        form.addRow("Job name:", self.fld_job_name)

        self.spin_cpus = QSpinBox()
        self.spin_cpus.setRange(1, 256)
        self.spin_cpus.setValue(4)
        self.spin_cpus.setMaximumWidth(100)
        self.spin_cpus.setToolTip("Number of CPU cores for the Abaqus solver.")
        form.addRow("CPUs:", self.spin_cpus)

        # Parallelisation: we only expose `cpus` for now. Abaqus offers
        # several parallelisation strategies (domain decomposition, loop
        # parallel, thread-only, ...) but the choice is non-trivial and
        # warrants its own UI later. For now we let Abaqus pick its own
        # defaults given `cpus`.

        # Workdir + browse
        wd_row = QHBoxLayout()
        self.fld_workdir = QLineEdit(self._get_prefs().default_workdir)
        self.fld_workdir.setToolTip(
            "Folder where the .inp, .odb, .log and .msg files will be written.\n"
            "Defaults to the value in Preferences; override per-job here."
        )
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_workdir)
        wd_row.addWidget(self.fld_workdir, stretch=1)
        wd_row.addWidget(btn_browse)
        wd_w = QWidget(); wd_w.setLayout(wd_row)
        form.addRow("Working directory:", wd_w)

        # Wire change handlers
        self.fld_job_name.textChanged.connect(self._on_change)
        self.spin_cpus.valueChanged.connect(self._on_change)
        self.fld_workdir.textChanged.connect(self._on_change)
        return g

    def _build_run_controls(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)

        self.btn_generate = QPushButton("Generate command (dry-run)")
        self.btn_generate.setToolTip(
            "Build the exact subprocess command that would be passed to\n"
            "ABQ.run_simul, plus the serialised model_params dict. No Abaqus\n"
            "process is launched."
        )
        self.btn_generate.clicked.connect(self._dry_run)
        row.addWidget(self.btn_generate)

        self.btn_run = QPushButton("Run Abaqus")
        self.btn_run.setToolTip(
            "Launch the Abaqus generator script with the current profile.\n"
            "Output is streamed live below; the GUI remains responsive."
        )
        self.btn_run.clicked.connect(self._run_abaqus)
        row.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip(
            "Kill the running Abaqus process. Use with care — the .odb\n"
            "may end up corrupt or partial."
        )
        self.btn_cancel.clicked.connect(self._cancel_run)
        row.addWidget(self.btn_cancel)

        self.btn_copy = QPushButton("Copy output")
        self.btn_copy.setToolTip("Copy the entire output panel to clipboard.")
        self.btn_copy.clicked.connect(self._copy_output)
        row.addStretch()
        row.addWidget(self.btn_copy)
        return bar

    def _build_output_panel(self) -> QGroupBox:
        g = QGroupBox("Output")
        v = QVBoxLayout(g)
        v.addWidget(_section_header(
            "Dry-run command preview, or live Abaqus log when running"
        ))

        self.txt_output = QPlainTextEdit()
        self.txt_output.setReadOnly(True)
        # Monospace font: makes the command and the repr() of params more
        # legible (alignment of brackets, quotes...). The live Abaqus log
        # also benefits — its output is column-aligned in places.
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.txt_output.setFont(mono)
        self.txt_output.setPlaceholderText(
            "• Click 'Generate command (dry-run)' to preview what would be\n"
            "  sent to Abaqus.\n"
            "• Click 'Run Abaqus' to launch the generator and stream its\n"
            "  output here in real time."
        )
        v.addWidget(self.txt_output)
        return g

    # =====================================================================
    # Actions
    # =====================================================================
    def _browse_workdir(self):
        cur = self.fld_workdir.text() or self._get_prefs().default_workdir
        path = QFileDialog.getExistingDirectory(
            self, "Pick working directory", cur or str(Path.home()),
        )
        if path:
            self.fld_workdir.setText(path)

    def _on_change(self, *_):
        self.jobChanged.emit()

    def _copy_output(self):
        QGuiApplication.clipboard().setText(self.txt_output.toPlainText())

    def _dry_run(self):
        """Build the command and the model_params repr, then display them.
        Mirrors ABQ.run_simul's command construction (without subprocess.run)."""
        prefs = self._get_prefs()

        job_name = self.fld_job_name.text().strip() or "Cutting_job"
        cpus     = int(self.spin_cpus.value())
        workdir  = self.fld_workdir.text().strip() or prefs.default_workdir

        # The model_params dict that abq_odb_generator.py will receive.
        model_params = self.cfg.to_params_dict()

        # The run_params dict ABQ.run_simul writes into. We let Abaqus
        # choose the parallelisation strategy (domain decomposition vs
        # loop parallel etc.) on its own for now.
        run_params = {"cpus": cpus, "job_name": job_name}

        # Reproduce ABQ.run_simul's command list literally.
        cmd = [
            prefs.abaqus_cmd,
            "cae",
            f"noGUI={prefs.abaqus_script}",
            "--",
            "--model_cfg",
            repr(model_params),
            "--run_cfg",
            repr(run_params),
        ]

        # Pretty-print the output panel with clearly separated blocks.
        lines = []
        lines.append("=" * 72)
        lines.append("DRY-RUN — nothing was launched")
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"Working directory:  {workdir}")
        lines.append(f"Job name:           {job_name}")
        lines.append(f"CPUs:               {cpus}")
        lines.append("")
        lines.append(f"Abaqus command:     {prefs.abaqus_cmd}")
        lines.append(f"Generator script:   {prefs.abaqus_script}")
        lines.append("")

        # Warn if the paths look obviously wrong (don't exist locally). We
        # don't block — the user might be staging files on another machine —
        # but it's the most common dry-run failure to catch early.
        warnings = []
        for label, p in (
            ("Abaqus command",   prefs.abaqus_cmd),
            ("Generator script", prefs.abaqus_script),
            ("Working directory", workdir),
        ):
            if not Path(p).exists():
                warnings.append(f"  ! {label} not found on disk: {p}")
        if warnings:
            lines.append("WARNINGS (paths not found on this machine):")
            lines.extend(warnings)
            lines.append("")

        lines.append("-" * 72)
        lines.append("subprocess command (as a Python list):")
        lines.append("-" * 72)
        # Format cmd as a multi-line repr so it's readable
        lines.append("[")
        for arg in cmd:
            lines.append(f"    {arg!r},")
        lines.append("]")
        lines.append("")

        lines.append("-" * 72)
        lines.append("Equivalent shell command (cwd = working directory):")
        lines.append("-" * 72)
        # For the shell version, quote each arg if it contains spaces / special
        # characters. Simple heuristic — good enough for visual inspection,
        # not safe for actual exec.
        shell_args = []
        for arg in cmd:
            if any(c in arg for c in ' \t"\'<>|&'):
                shell_args.append(f'"{arg}"')
            else:
                shell_args.append(arg)
        lines.append(" ".join(shell_args))
        lines.append("")

        lines.append("-" * 72)
        lines.append("model_params (what abq_odb_generator.py will receive):")
        lines.append("-" * 72)
        lines.append(repr(model_params))
        lines.append("")

        lines.append("-" * 72)
        lines.append("run_params:")
        lines.append("-" * 72)
        lines.append(repr(run_params))
        lines.append("")

        self.txt_output.setPlainText("\n".join(lines))

    # =====================================================================
    # Real Abaqus execution (QProcess, non-blocking, streamed output)
    # =====================================================================
    def _run_abaqus(self):
        """Launch the Abaqus generator script in a child process. Output
        (stdout + stderr merged) is streamed live into the output panel
        as it arrives. The GUI thread remains responsive throughout.
        """
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            # Already running — protect the user from launching twice.
            QMessageBox.warning(
                self, "Already running",
                "An Abaqus run is already in progress. Cancel it first if\n"
                "you want to start a new one.",
            )
            return

        prefs = self._get_prefs()
        job_name = self.fld_job_name.text().strip() or "Cutting_job"
        cpus     = int(self.spin_cpus.value())
        workdir  = self.fld_workdir.text().strip() or prefs.default_workdir

        # Sanity checks before launching — we want the user to see the
        # error as a dialog, not as a cryptic QProcess errorOccurred signal.
        problems = []
        if not Path(prefs.abaqus_cmd).exists():
            problems.append(f"Abaqus command not found: {prefs.abaqus_cmd}")
        if not Path(prefs.abaqus_script).exists():
            problems.append(f"Generator script not found: {prefs.abaqus_script}")
        wd = Path(workdir)
        try:
            wd.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            problems.append(f"Cannot create working directory '{workdir}': {e}")
        if problems:
            QMessageBox.critical(
                self, "Cannot launch Abaqus",
                "Please fix these issues before running:\n\n• "
                + "\n• ".join(problems)
                + "\n\nYou can edit the Abaqus command and generator path\n"
                  "in the Preferences dialog (top-left menu).",
            )
            return

        # Build the args (same construction as the dry-run + ABQ.run_simul)
        model_params = self.cfg.to_params_dict()
        run_params   = {"cpus": cpus, "job_name": job_name}
        args = [
            "cae",
            f"noGUI={prefs.abaqus_script}",
            "--",
            "--model_cfg",
            repr(model_params),
            "--run_cfg",
            repr(run_params),
        ]

        # Header in the output panel — replaces the dry-run text.
        header = [
            "=" * 72,
            f"RUNNING Abaqus — job: {job_name}",
            "=" * 72,
            f"Working directory:  {workdir}",
            f"CPUs:               {cpus}",
            f"Abaqus command:     {prefs.abaqus_cmd}",
            f"Generator script:   {prefs.abaqus_script}",
            "-" * 72,
            "Live output (stdout + stderr):",
            "-" * 72,
            "",
        ]
        self.txt_output.setPlainText("\n".join(header))

        # Create and configure the QProcess
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(wd))
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        # On Windows, Abaqus uses cp1252 for its console output. We read raw
        # bytes from the process and decode in the slot; see _on_proc_output.
        self._proc.readyReadStandardOutput.connect(self._on_proc_output)
        self._proc.finished.connect(self._on_proc_finished)
        self._proc.errorOccurred.connect(self._on_proc_error)

        # Disable Run + enable Cancel during execution
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running…")
        self.btn_cancel.setEnabled(True)
        self.btn_generate.setEnabled(False)

        # Launch
        self._proc.start(prefs.abaqus_cmd, args)

    def _cancel_run(self):
        """Forcefully terminate the running Abaqus process. Use sparingly:
        the .odb may be left in an incomplete state."""
        if self._proc is None or self._proc.state() == QProcess.NotRunning:
            return
        reply = QMessageBox.question(
            self, "Cancel run?",
            "Kill the running Abaqus process now?\n"
            "The .odb file may be incomplete or corrupt.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        # On Windows, terminate() sends WM_CLOSE which Abaqus may ignore;
        # kill() is more reliable. Try terminate first, then kill if it
        # hasn't exited within 2 seconds.
        self._proc.terminate()
        if not self._proc.waitForFinished(2000):
            self._proc.kill()
        self._append_output("\n\n[CANCELLED by user]\n")

    def _on_proc_output(self):
        """Slot connected to QProcess.readyReadStandardOutput. Reads
        whatever bytes are available and appends them to the output
        panel as text."""
        if self._proc is None:
            return
        raw = bytes(self._proc.readAllStandardOutput())
        try:
            text = raw.decode("cp1252", errors="replace")
        except Exception:
            text = raw.decode("utf-8", errors="replace")
        self._append_output(text)

    def _on_proc_finished(self, exit_code: int, exit_status):
        """Slot connected to QProcess.finished. Restores button states
        and prints a summary footer."""
        # Flush any remaining buffered output before the footer
        if self._proc is not None:
            tail = bytes(self._proc.readAllStandardOutput())
            if tail:
                self._append_output(tail.decode("cp1252", errors="replace"))

        if exit_status == QProcess.NormalExit and exit_code == 0:
            footer = "\n" + "=" * 72 + "\n[DONE] Abaqus exited cleanly (code 0)\n"
        else:
            footer = (
                "\n" + "=" * 72 +
                f"\n[FAILED] Abaqus exit code: {exit_code}, status: {exit_status}\n"
            )
            # Hint: Abaqus often writes the actual error to STDOUT before
            # exiting non-zero, so the output panel above already contains
            # the diagnostic; we don't need to pop a dialog.
        self._append_output(footer)

        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Abaqus")
        self.btn_cancel.setEnabled(False)
        self.btn_generate.setEnabled(True)
        self._proc = None

    def _on_proc_error(self, error):
        """Slot connected to QProcess.errorOccurred. Fires for issues
        like 'process failed to start' (executable missing, permission
        denied, etc.) — distinct from a non-zero exit which is handled
        in _on_proc_finished."""
        # Map QProcess.ProcessError to a human label
        labels = {
            QProcess.FailedToStart: "Failed to start the Abaqus executable.",
            QProcess.Crashed:       "Abaqus crashed.",
            QProcess.Timedout:      "Abaqus timed out.",
            QProcess.WriteError:    "I/O write error to the Abaqus process.",
            QProcess.ReadError:     "I/O read error from the Abaqus process.",
        }
        msg = labels.get(error, f"QProcess error: {error}")
        self._append_output(f"\n[ERROR] {msg}\n")

    def _append_output(self, text: str):
        """Append text to the output panel and keep the view scrolled to
        the bottom — so the user sees the latest log line without having
        to scroll manually."""
        if not text:
            return
        cursor = self.txt_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.txt_output.setTextCursor(cursor)
        self.txt_output.ensureCursorVisible()

    # =====================================================================
    # External hooks (round-trip not yet implemented for Job fields —
    # they don't live in ModelConfig yet)
    # =====================================================================
    def apply_from_cfg(self):
        """No-op for now: job parameters aren't part of the .acpf profile.
        Workdir defaults to the user-level Preferences each time."""
        pass
