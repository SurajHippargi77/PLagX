"""
PLagX CLI – Cross-Language Plagiarism Intelligence

Usage:
    python -m plagx download              Download reference documents from Wikipedia
    python -m plagx check <file.txt>      Check a text file for cross-lingual plagiarism
    python -m plagx check <file.pdf>      Check a PDF for cross-lingual plagiarism
    python -m plagx check --text "..."    Check inline text
    python -m plagx status                Show downloaded reference documents
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Download command ─────────────────────────────────────────────────────────

def cmd_download(args):
    """Download reference documents from Wikipedia for all 5 languages."""
    from plagx.downloader import download_documents
    from plagx.config import LANGUAGES

    console.print(Panel(
        "[bold cyan]PLagX – Downloading Reference Documents[/bold cyan]\n"
        "Languages: " + ", ".join(f"{v['name']} ({k})" for k, v in LANGUAGES.items()),
        box=box.DOUBLE,
    ))

    with console.status("[bold green]Downloading from Wikipedia..."):
        result = download_documents(force=args.force)

    total = 0
    table = Table(title="Downloaded Documents", box=box.ROUNDED)
    table.add_column("Language", style="cyan")
    table.add_column("Files", style="green")
    table.add_column("Count", justify="right")

    for lang_code, files in result.items():
        lang_name = LANGUAGES[lang_code]["name"]
        file_list = ", ".join(f.stem for f in files) if files else "(none)"
        table.add_row(lang_name, file_list, str(len(files)))
        total += len(files)

    console.print(table)
    console.print(f"\n[bold green]Total: {total} documents downloaded.[/bold green]")


# ── Status command ───────────────────────────────────────────────────────────

def cmd_status(args):
    """Show current reference documents."""
    from plagx.downloader import list_downloaded
    from plagx.config import LANGUAGES, DATA_DIR

    console.print(Panel("[bold cyan]PLagX – Reference Document Status[/bold cyan]", box=box.DOUBLE))
    console.print(f"Data directory: {DATA_DIR}\n")

    downloaded = list_downloaded()
    if not downloaded:
        console.print("[yellow]No reference documents found. Run: python -m plagx download[/yellow]")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("Language", style="cyan")
    table.add_column("Documents", style="green")
    table.add_column("Total Size", justify="right")

    for lang_code, files in downloaded.items():
        lang_name = LANGUAGES.get(lang_code, {}).get("name", lang_code)
        names = [f.stem for f in files]
        size = sum(f.stat().st_size for f in files)
        size_str = f"{size / 1024:.1f} KB"
        table.add_row(lang_name, ", ".join(names), size_str)

    console.print(table)


# ── Check command ────────────────────────────────────────────────────────────

def cmd_check(args):
    """Run plagiarism check on input text or file."""
    from plagx.detector import detect, detect_from_pdf
    from plagx.config import LANGUAGES

    # Get input text
    if args.text:
        input_text = args.text
        source_label = "(inline text)"
    elif args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            console.print(f"[red]File not found: {args.file}[/red]")
            sys.exit(1)

        if fpath.suffix.lower() == ".pdf":
            console.print(Panel("[bold cyan]PLagX – Checking PDF for plagiarism[/bold cyan]", box=box.DOUBLE))
            with console.status("[bold green]Analyzing PDF..."):
                report = detect_from_pdf(str(fpath), threshold=args.threshold)
            _print_report(report, str(fpath))
            return
        else:
            input_text = fpath.read_text(encoding="utf-8", errors="replace")
            source_label = str(fpath)
    else:
        console.print("[red]Provide --text or --file[/red]")
        sys.exit(1)

    if not input_text.strip():
        console.print("[red]Input text is empty.[/red]")
        sys.exit(1)

    # Parse language filter
    lang_filter = None
    if args.languages:
        lang_filter = [l.strip() for l in args.languages.split(",")]

    console.print(Panel(
        "[bold cyan]PLagX – Cross-Language Plagiarism Check[/bold cyan]\n"
        f"Source: {source_label}",
        box=box.DOUBLE,
    ))

    with console.status("[bold green]Running plagiarism analysis... (this may take a minute on first run)"):
        report = detect(input_text, lang_codes=lang_filter, threshold=args.threshold)

    _print_report(report, source_label)


def _print_report(report, source_label: str):
    """Pretty-print a PlagiarismReport."""
    from plagx.config import LANGUAGES

    # ── Overall result ────────────────────────────────────────────────────
    pct = report.overall_plagiarism_pct
    if pct >= 50:
        color = "red"
        verdict = "HIGH PLAGIARISM DETECTED"
    elif pct >= 25:
        color = "yellow"
        verdict = "MODERATE PLAGIARISM DETECTED"
    elif pct > 0:
        color = "cyan"
        verdict = "LOW PLAGIARISM DETECTED"
    else:
        color = "green"
        verdict = "NO PLAGIARISM DETECTED"

    console.print()
    console.print(Panel(
        f"[bold {color}]{verdict}[/bold {color}]\n\n"
        f"Overall plagiarism: [bold]{pct:.1f}%[/bold]\n"
        f"Input sentences analyzed: {report.total_input_sentences}\n"
        f"Sentences flagged: {len(report.flagged_sentences)}\n"
        f"Threshold used: {report.threshold_used:.2f}",
        title="[bold]RESULT[/bold]",
        box=box.HEAVY,
    ))

    # ── Per-language breakdown ────────────────────────────────────────────
    lang_pct = report.per_language_pct()
    if lang_pct:
        table = Table(title="Plagiarism by Language", box=box.ROUNDED)
        table.add_column("Language", style="cyan")
        table.add_column("Match %", justify="right", style="bold")
        table.add_column("Matched Sentences", justify="right")
        table.add_column("Documents Checked", justify="right")

        for code, pct_val in sorted(lang_pct.items(), key=lambda x: -x[1]):
            lr = report.languages[code]
            lang_name = LANGUAGES.get(code, {}).get("name", code)
            n_docs = len(lr.documents)
            n_matches = lr.total_matches
            style = "red" if pct_val >= 30 else ("yellow" if pct_val >= 10 else "green")
            table.add_row(
                lang_name,
                f"[{style}]{pct_val:.1f}%[/{style}]",
                str(n_matches),
                str(n_docs),
            )
        console.print(table)

    # ── Per-document breakdown ────────────────────────────────────────────
    all_doc_results = []
    for lr in report.languages.values():
        for d in lr.documents:
            if d.match_count > 0:
                all_doc_results.append(d)

    if all_doc_results:
        all_doc_results.sort(key=lambda d: -d.avg_score)

        table = Table(title="Top Matching Documents", box=box.ROUNDED)
        table.add_column("Document", style="cyan")
        table.add_column("Language", style="magenta")
        table.add_column("Matches", justify="right")
        table.add_column("Avg Score", justify="right", style="bold")

        for d in all_doc_results[:10]:  # top 10
            table.add_row(
                d.filename,
                d.language_name,
                str(d.match_count),
                f"{d.avg_score:.3f}",
            )
        console.print(table)

    # ── Top sentence matches ──────────────────────────────────────────────
    all_matches = report.all_matches
    if all_matches:
        all_matches_sorted = sorted(all_matches, key=lambda m: -m.combined_score)

        console.print()
        console.print("[bold underline]Top Sentence Matches[/bold underline]")
        console.print()

        for i, m in enumerate(all_matches_sorted[:15], 1):  # top 15
            sem_bar = "█" * int(m.semantic_score * 20)
            tfidf_bar = "█" * int(m.tfidf_score * 20)

            console.print(f"[bold cyan]Match #{i}[/bold cyan]  "
                          f"[bold]Combined: {m.combined_score:.3f}[/bold]  "
                          f"({m.language_name} / {m.document})")
            console.print(f"  [green]EN Input:[/green]    {m.query_sentence[:120]}")
            console.print(f"  [yellow]Original:[/yellow]    {m.ref_sentence_original[:120]}")
            console.print(f"  [blue]Translated:[/blue]  {m.ref_sentence_english[:120]}")
            console.print(f"  Semantic: {m.semantic_score:.3f} {sem_bar}")
            console.print(f"  TF-IDF:   {m.tfidf_score:.3f} {tfidf_bar}")
            console.print()

    if not all_matches:
        console.print("\n[green]No matches found above threshold. Text appears original.[/green]")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="plagx",
        description="PLagX – Cross-Language Plagiarism Intelligence",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # download
    dl = sub.add_parser("download", help="Download reference documents from Wikipedia")
    dl.add_argument("--force", action="store_true", help="Re-download even if files exist")

    # status
    sub.add_parser("status", help="Show downloaded reference documents")

    # check
    chk = sub.add_parser("check", help="Check text/PDF for cross-lingual plagiarism")
    chk.add_argument("--file", "-f", help="Path to .txt or .pdf file to check")
    chk.add_argument("--text", "-t", help="Inline English text to check")
    chk.add_argument("--languages", "-l", help="Comma-separated language codes (e.g. nl,ja,de)")
    chk.add_argument("--threshold", type=float, default=None, help="Override similarity threshold (0-1)")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "download":
        cmd_download(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "check":
        cmd_check(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
