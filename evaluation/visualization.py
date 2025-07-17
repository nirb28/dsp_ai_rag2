"""
Visualization utilities for RAG evaluation results.
"""
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from evaluation.base import EvaluationResult, BaseEvaluator

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """Visualization tools for evaluation results."""
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("evaluation/results/figures")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_comparison(
        self,
        results: List[EvaluationResult],
        metric_name: str,
        group_by: str = "configuration_name",
        title: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Plot comparison of a specific metric across different groups.
        
        Args:
            results: List of EvaluationResult objects
            metric_name: Name of the metric to plot
            group_by: Metadata field to group results by
            title: Plot title (optional)
            filename: Filename for saved plot (optional)
            
        Returns:
            Path to saved figure
        """
        # Extract data
        data = []
        
        for result in results:
            if result.metric_name == metric_name and metric_name in result.value:
                metric_value = result.value[metric_name]
                group = result.metadata.get(group_by, "unknown")
                data.append({"value": metric_value, group_by: group})
        
        if not data:
            logger.warning(f"No data found for metric '{metric_name}'")
            return None
            
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=group_by, y="value", data=df)
        
        # Add labels and title
        plt.xlabel(group_by.replace("_", " ").title())
        plt.ylabel(metric_name.replace("_", " ").title())
        
        if title:
            plt.title(title)
        else:
            plt.title(f"{metric_name.replace('_', ' ').title()} by {group_by.replace('_', ' ').title()}")
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.01,
                   f'{p.get_height():.3f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure
        if not filename:
            filename = f"{metric_name}_by_{group_by}.png"
            
        filepath = self.save_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        logger.info(f"Saved metrics comparison plot to {filepath}")
        return str(filepath)
    
    def plot_retrieval_metrics(
        self,
        results: List[EvaluationResult],
        title: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Plot retrieval metrics (precision, recall, F1, MRR) for different configurations.
        
        Args:
            results: List of EvaluationResult objects
            title: Plot title (optional)
            filename: Filename for saved plot (optional)
            
        Returns:
            Path to saved figure
        """
        # Extract data
        data = []
        metric_types = ["precision", "recall", "f1", "mrr"]
        
        for result in results:
            if result.metric_name in metric_types:
                metric_value = result.value
                config_name = result.metadata.get("configuration_name", "unknown")
                k_value = result.metadata.get("k", "unknown")
                data.append({
                    "metric": result.metric_name,
                    "value": metric_value,
                    "configuration": config_name,
                    "k": k_value
                })
        
        if not data:
            logger.warning("No retrieval metrics found in results")
            return None
            
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # If multiple k values, show them as different groups
        if len(df["k"].unique()) > 1:
            ax = sns.catplot(
                x="configuration", 
                y="value", 
                hue="k", 
                col="metric",
                kind="bar", 
                data=df,
                col_wrap=2,
                height=4,
                aspect=1.2,
                legend_out=False
            )
            plt.subplots_adjust(top=0.9)
            
            if title:
                plt.suptitle(title)
            else:
                plt.suptitle("Retrieval Metrics by Configuration and k")
                
        else:
            ax = sns.catplot(
                x="configuration", 
                y="value", 
                hue="metric", 
                kind="bar", 
                data=df,
                height=6,
                aspect=1.5,
                legend_out=False
            )
            
            if title:
                plt.title(title)
            else:
                plt.title("Retrieval Metrics by Configuration")
        
        # Save figure
        if not filename:
            filename = "retrieval_metrics.png"
            
        filepath = self.save_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        logger.info(f"Saved retrieval metrics plot to {filepath}")
        return str(filepath)
    
    def plot_query_performance(
        self,
        results: List[EvaluationResult],
        title: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Plot query performance metrics across configurations.
        
        Args:
            results: List of EvaluationResult objects
            title: Plot title (optional)
            filename: Filename for saved plot (optional)
            
        Returns:
            Path to saved figure
        """
        # Extract query performance data
        data = []
        
        for result in results:
            if result.metric_name == "configuration_benchmark":
                config_name = result.metadata.get("configuration_name", "unknown")
                
                # Extract key metrics
                avg_time = result.value.get("avg_query_time", 0)
                p95_time = result.value.get("p95_query_time", 0)
                qps = result.value.get("queries_per_second", 0)
                
                data.append({
                    "configuration": config_name,
                    "metric": "Average Query Time (s)",
                    "value": avg_time
                })
                
                data.append({
                    "configuration": config_name,
                    "metric": "P95 Query Time (s)",
                    "value": p95_time
                })
                
                data.append({
                    "configuration": config_name,
                    "metric": "Queries Per Second",
                    "value": qps
                })
        
        if not data:
            logger.warning("No query performance metrics found in results")
            return None
            
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create separate plots for timing metrics and QPS
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Timing metrics
        timing_df = df[df["metric"].isin(["Average Query Time (s)", "P95 Query Time (s)"])]
        sns.barplot(x="configuration", y="value", hue="metric", data=timing_df, ax=axes[0])
        axes[0].set_title("Query Response Time by Configuration")
        axes[0].set_ylabel("Time (seconds)")
        
        # QPS metrics
        qps_df = df[df["metric"] == "Queries Per Second"]
        sns.barplot(x="configuration", y="value", data=qps_df, ax=axes[1], color="green")
        axes[1].set_title("Query Throughput by Configuration")
        axes[1].set_ylabel("Queries Per Second")
        
        if title:
            fig.suptitle(title, fontsize=16, y=1.05)
        
        plt.tight_layout()
        
        # Save figure
        if not filename:
            filename = "query_performance.png"
            
        filepath = self.save_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        logger.info(f"Saved query performance plot to {filepath}")
        return str(filepath)
    
    def plot_query_evaluation_metrics(
        self,
        results: List[EvaluationResult],
        title: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Plot query evaluation metrics (correctness, relevance, completeness).
        
        Args:
            results: List of EvaluationResult objects
            title: Plot title (optional)
            filename: Filename for saved plot (optional)
            
        Returns:
            Path to saved figure
        """
        # Extract data for average metrics
        data = []
        metric_names = [
            "avg_answer_correctness",
            "avg_answer_relevance",
            "avg_answer_completeness",
            "avg_semantic_similarity"
        ]
        
        for result in results:
            if result.metric_name in metric_names:
                metric_type = result.metric_name.replace("avg_", "")
                config_name = result.metadata.get("configuration_name", "unknown")
                
                data.append({
                    "metric": metric_type,
                    "configuration": config_name,
                    "value": result.value
                })
        
        if not data:
            logger.warning("No query evaluation metrics found in results")
            return None
            
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        if len(df["configuration"].unique()) > 1:
            # Multiple configurations - group by configuration
            ax = sns.catplot(
                x="metric",
                y="value",
                hue="configuration",
                kind="bar",
                data=df,
                height=6,
                aspect=1.5
            )
            
            plt.xticks(rotation=45)
            
        else:
            # Single configuration - simple bar chart
            ax = sns.barplot(
                x="metric",
                y="value",
                data=df
            )
            
            # Add value labels on top of bars
            for i, p in enumerate(ax.patches):
                ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.01,
                       f'{p.get_height():.3f}', ha='center')
            
            plt.xticks(rotation=45)
        
        if title:
            plt.title(title)
        else:
            plt.title("Answer Quality Metrics")
            
        plt.ylabel("Score (0-1)")
        plt.tight_layout()
        
        # Save figure
        if not filename:
            filename = "query_evaluation_metrics.png"
            
        filepath = self.save_dir / filename
        plt.savefig(filepath)
        plt.close()
        
        logger.info(f"Saved query evaluation metrics plot to {filepath}")
        return str(filepath)
    
    def create_dashboard(
        self,
        results_files: List[str],
        output_file: Optional[str] = None
    ) -> str:
        """
        Create an HTML dashboard with all evaluation results.
        
        Args:
            results_files: List of paths to results JSON files
            output_file: Path for output HTML file
            
        Returns:
            Path to saved HTML dashboard
        """
        # Load all results
        all_results = []
        
        for filepath in results_files:
            try:
                evaluator = BaseEvaluator("temp", "")  # Temporary instance just for loading
                results = evaluator.load_results(filepath)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error loading results from {filepath}: {e}")
        
        if not all_results:
            logger.warning("No results loaded for dashboard")
            return None
        
        # Convert to dataframe for easier processing
        records = []
        
        for result in all_results:
            record = {
                "metric_name": result.metric_name,
                "timestamp": result.timestamp
            }
            
            # Handle different value types
            if isinstance(result.value, dict):
                for k, v in result.value.items():
                    record[f"value_{k}"] = v
            else:
                record["value"] = result.value
            
            # Add metadata
            for k, v in result.metadata.items():
                if not isinstance(v, (dict, list)):
                    record[f"meta_{k}"] = v
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Generate HTML with plots
        from jinja2 import Template
        
        # Create simple plots for dashboard
        plot_paths = []
        
        # Retrieval metrics if available
        retrieval_metrics = [r for r in all_results if r.metric_name in ["precision", "recall", "f1", "mrr"]]
        if retrieval_metrics:
            plot_path = self.plot_retrieval_metrics(retrieval_metrics, filename="dashboard_retrieval.png")
            if plot_path:
                plot_paths.append(("Retrieval Performance", Path(plot_path).name))
        
        # Query evaluation metrics if available
        query_metrics = [r for r in all_results if r.metric_name.startswith("avg_")]
        if query_metrics:
            plot_path = self.plot_query_evaluation_metrics(query_metrics, filename="dashboard_query_eval.png")
            if plot_path:
                plot_paths.append(("Answer Quality", Path(plot_path).name))
        
        # Query performance metrics if available
        perf_metrics = [r for r in all_results if r.metric_name == "configuration_benchmark"]
        if perf_metrics:
            plot_path = self.plot_query_performance(perf_metrics, filename="dashboard_performance.png")
            if plot_path:
                plot_paths.append(("Query Performance", Path(plot_path).name))
        
        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin-top: 10px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .plot-container { margin-top: 20px; margin-bottom: 30px; }
                .plot-image { max-width: 100%; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>RAG Evaluation Dashboard</h1>
            <p>Generated on: {{ timestamp }}</p>
            
            {% for section in plot_sections %}
            <h2>{{ section[0] }}</h2>
            <div class="plot-container">
                <img class="plot-image" src="{{ section[1] }}" alt="{{ section[0] }}">
            </div>
            {% endfor %}
            
            <h2>All Metrics</h2>
            <table>
                <tr>
                    {% for col in columns %}
                    <th>{{ col }}</th>
                    {% endfor %}
                </tr>
                {% for row in rows %}
                <tr>
                    {% for col in columns %}
                    <td>{{ row[col] }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """
        
        # Prepare table data for HTML
        display_cols = [
            col for col in df.columns 
            if not col.startswith("value_") or col in ["value", "metric_name", "timestamp"] 
            or col.startswith("meta_configuration")
        ]
        
        template = Template(html_template)
        html_content = template.render(
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            plot_sections=plot_paths,
            columns=display_cols,
            rows=df[display_cols].to_dict(orient="records")
        )
        
        # Save HTML file
        if not output_file:
            output_file = "rag_evaluation_dashboard.html"
            
        filepath = self.save_dir / output_file
        
        with open(filepath, "w") as f:
            f.write(html_content)
        
        logger.info(f"Created evaluation dashboard at {filepath}")
        return str(filepath)
