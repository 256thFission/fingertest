#!/usr/bin/env python3
"""
Experiment tracking and automatic documentation management.
"""

import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Manages experiment documentation and status tracking."""

    def __init__(self, config: "ExperimentConfig"):
        self.config = config
        self.exp_dir = Path("experiments")
        self.exp_doc_path = self._get_doc_path()
        self.log_path = self.exp_dir / "README.md"

    def _get_doc_path(self) -> Path:
        """Get experiment doc path."""
        return self.exp_dir / f"{self.config.experiment.id}_{self.config.experiment.name}.md"

    def create_experiment_doc(self):
        """Create experiment doc from simplified template."""
        if self.exp_doc_path.exists():
            logger.info(f"Experiment doc already exists: {self.exp_doc_path}")
            return

        template = self._get_simple_template()
        self.exp_doc_path.write_text(template)
        logger.info(f"Created experiment doc: {self.exp_doc_path}")

    def _get_simple_template(self) -> str:
        """Generate simple experiment template."""
        config = self.config
        date = datetime.now().strftime("%Y-%m-%d")

        return f"""# Experiment {config.experiment.id}: {config.experiment.name}

**Date:** {date}
**Status:** {config.experiment.status}
**Parent:** {config.experiment.parent_experiment or "None"}

## Hypothesis

{config.experiment.hypothesis}

## Description

{config.experiment.description}

## Expected Results

{self._format_expected_results()}

## Configuration

**Config file:** `configs/experiments/{config.experiment.id}_{config.experiment.name}.yaml`

**Key parameters:**
- Model: {config.model.base_model}
- Data version: {config.data.version}
- Loss: {self._get_loss_info()}
- Batch size: {self._get_batch_size()}
- Epochs: {self._get_epochs()}
- Whitening: {config.evaluation.use_whitening}

## Results

*Results will be automatically updated after training completes.*

## Notes

"""

    def _format_expected_results(self) -> str:
        """Format expected results dict."""
        if not self.config.experiment.expected_results:
            return "- TBD"

        lines = []
        for key, value in self.config.experiment.expected_results.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _get_loss_info(self) -> str:
        """Get loss configuration info."""
        if self.config.baseline_training:
            loss = self.config.baseline_training.loss
            return f"{loss.type} (scale={loss.scale}, temp={loss.temperature})"
        elif self.config.triplet_training:
            loss = self.config.triplet_training.loss
            return f"{loss.type} (margin={loss.margin})"
        else:
            return "N/A"

    def _get_batch_size(self) -> str:
        """Get batch size info."""
        if self.config.baseline_training:
            return str(self.config.baseline_training.batch_size)
        elif self.config.triplet_training:
            return str(self.config.triplet_training.batch_size)
        else:
            return "N/A"

    def _get_epochs(self) -> str:
        """Get epochs info."""
        if self.config.baseline_training:
            return str(self.config.baseline_training.num_epochs)
        elif self.config.triplet_training:
            return str(self.config.triplet_training.num_epochs)
        elif self.config.loop:
            return f"{self.config.loop.num_iterations} iterations"
        else:
            return "N/A"

    def update_status(self, status: str):
        """Update experiment status in doc."""
        if not self.exp_doc_path.exists():
            return

        content = self.exp_doc_path.read_text()

        # Update status line
        content = re.sub(
            r'\*\*Status:\*\* \w+',
            f'**Status:** {status}',
            content
        )

        self.exp_doc_path.write_text(content)
        logger.info(f"Updated experiment status: {status}")

        # Update experiment log
        self._update_log_status(status)

    def log_start(self):
        """Log experiment start."""
        self.update_status("running")
        logger.info(f"Experiment {self.config.experiment.id} started")

    def log_results(self, metrics: Dict[str, Any], wandb_url: Optional[str] = None):
        """Update experiment doc with results."""
        if not self.exp_doc_path.exists():
            logger.warning("Experiment doc doesn't exist, cannot update results")
            return

        content = self.exp_doc_path.read_text()

        # Generate results section
        results_section = self._generate_results_section(metrics, wandb_url)

        # Replace results section
        content = self._replace_section(content, "## Results", results_section)

        self.exp_doc_path.write_text(content)
        logger.info("Updated experiment doc with results")

        # Update experiment log
        self._update_log_results(metrics)

    def _generate_results_section(self, metrics: Dict[str, Any], wandb_url: Optional[str]) -> str:
        """Generate results section markdown."""
        eer = metrics.get("eer", 0)
        roc_auc = metrics.get("roc_auc", 0)
        target_eer = self.config.evaluation.target_eer
        target_auc = self.config.evaluation.target_roc_auc

        # Calculate deltas if parent exists
        eer_delta = ""
        auc_delta = ""
        if self.config.experiment.parent_experiment:
            parent_metrics = self._get_parent_metrics()
            if parent_metrics:
                eer_change = eer - parent_metrics.get("eer", eer)
                auc_change = roc_auc - parent_metrics.get("roc_auc", roc_auc)
                eer_delta = f" ({eer_change:+.2%})"
                auc_delta = f" ({auc_change:+.4f})"

        section = f"""## Results

### Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **EER** | {eer:.2%}{eer_delta} | {target_eer:.2%} | {'' if eer <= target_eer else ''} |
| **ROC-AUC** | {roc_auc:.4f}{auc_delta} | {target_auc:.4f} | {'' if roc_auc >= target_auc else ''} |
| **EER Threshold** | {metrics.get('eer_threshold', 'N/A'):.4f} | - | - |
| **Accuracy @ EER** | {metrics.get('accuracy_at_eer', 'N/A'):.2%} | - | - |

"""

        # Add wandb link if available
        if wandb_url:
            section += f"**Wandb Run:** [{wandb_url}]({wandb_url})\n\n"

        # Add visualizations
        output_dir = Path(self.config.evaluation.output_dir)
        if output_dir.exists():
            section += "### Visualizations\n\n"
            for plot in ["roc_curve", "far_frr_curves", "score_distribution", "umap_visualization"]:
                plot_path = output_dir / f"{plot}.png"
                if plot_path.exists():
                    section += f"![{plot}]({plot_path})\n\n"

        return section

    def _replace_section(self, content: str, section_header: str, new_content: str) -> str:
        """Replace a markdown section."""
        # Find section start
        pattern = f"({re.escape(section_header)}.*?)(\n##|\Z)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            # Replace section
            return content[:match.start(1)] + new_content + content[match.start(2):]
        else:
            # Append section
            return content + "\n\n" + new_content

    def _get_parent_metrics(self) -> Optional[Dict[str, float]]:
        """Get metrics from parent experiment."""
        if not self.config.experiment.parent_experiment:
            return None

        try:
            parent_id = self.config.experiment.parent_experiment
            # Find parent experiment doc
            parent_docs = list(self.exp_dir.glob(f"{parent_id}_*.md"))
            if not parent_docs:
                return None

            # Extract metrics from parent doc
            content = parent_docs[0].read_text()

            # Parse EER and ROC-AUC from table
            eer_match = re.search(r'\| \*\*EER\*\* \| ([\d.]+)%', content)
            auc_match = re.search(r'\| \*\*ROC-AUC\*\* \| ([\d.]+)', content)

            if eer_match and auc_match:
                return {
                    "eer": float(eer_match.group(1)) / 100,
                    "roc_auc": float(auc_match.group(1)),
                }
        except Exception as e:
            logger.warning(f"Failed to get parent metrics: {e}")

        return None

    def _update_log_status(self, status: str):
        """Update experiments/README.md with status."""
        if not self.log_path.exists():
            return

        content = self.log_path.read_text()

        # Find or create row for this experiment
        exp_id = self.config.experiment.id
        row_pattern = f"\\| {exp_id} \\|.*"

        if re.search(row_pattern, content):
            # Update existing row - just update status column
            def replace_status(match):
                parts = match.group(0).split("|")
                # Status is typically in column 4 (after ID, date, name)
                if len(parts) > 4:
                    parts[4] = f" {self._status_emoji(status)} {status.title()} "
                return "|".join(parts)

            content = re.sub(row_pattern, replace_status, content)
        else:
            # Add new row
            new_row = self._create_log_row(status=status)
            content = self._insert_log_row(content, new_row)

        self.log_path.write_text(content)

    def _update_log_results(self, metrics: Dict[str, Any]):
        """Update experiments/README.md with results."""
        if not self.log_path.exists():
            return

        content = self.log_path.read_text()
        exp_id = self.config.experiment.id
        row_pattern = f"\\| {exp_id} \\|.*"

        # Update row with metrics
        new_row = self._create_log_row(status="complete", metrics=metrics)

        if re.search(row_pattern, content):
            content = re.sub(row_pattern, new_row, content)
        else:
            content = self._insert_log_row(content, new_row)

        self.log_path.write_text(content)

    def _create_log_row(self, status: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Create a log table row."""
        date = datetime.now().strftime("%Y-%m-%d")
        exp_id = self.config.experiment.id
        name = self.config.experiment.name

        eer = f"{metrics['eer']:.2%}" if metrics else "-"
        auc = f"{metrics['roc_auc']:.4f}" if metrics else "-"

        doc_link = f"[{name}]({self.exp_doc_path.name})"

        return f"| {exp_id} | {date} | {doc_link} | {self._status_emoji(status)} {status.title()} | {eer} | {auc} |"

    def _status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        return {
            "planning": "",
            "running": "",
            "complete": "",
            "failed": "",
        }.get(status.lower(), "")

    def _insert_log_row(self, content: str, new_row: str) -> str:
        """Insert new row into log table."""
        # Find table and insert after header
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("| ID |"):
                # Insert after separator line
                if i + 1 < len(lines):
                    lines.insert(i + 2, new_row)
                    break

        return "\n".join(lines)
