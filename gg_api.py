
from winners import run_winners
from nominees import run_nominees
from hosts import run_hosts
from awards import run_awards
from presenters import run_presenters
import json
import time
import sys

def _ensure_nltk():
    try:
        import nltk
        for pkg in ["punkt", "averaged_perceptron_tagger", "stopwords"]:
            try:
                nltk.data.find(f"tokenizers/{pkg}") if pkg == "punkt" else nltk.data.find(
                    f"taggers/{pkg}" if "tagger" in pkg else f"corpora/{pkg}"
                )
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:
        pass

def _ensure_spacy_model():
    try:
        import spacy
        from spacy.util import is_package
        try:

            spacy.load("en_core_web_sm")
            return
        except Exception:
            pass
        try:
            if spacy.__version__.startswith("2."):
                from spacy.cli import download
                download("en_core_web_sm==2.2.5")
            else:
                from spacy.cli import download
                download("en_core_web_sm")
        except Exception:
            pass
    except Exception:
        pass

AWARD_NAMES = [
    'cecil b. demille award', 'best motion picture - drama',
    'best performance by an actress in a motion picture - drama',
    'best performance by an actor in a motion picture - drama',
    'best motion picture - comedy or musical',
    'best performance by an actress in a motion picture - comedy or musical',
    'best performance by an actor in a motion picture - comedy or musical',
    'best animated feature film', 'best foreign language film',
    'best performance by an actress in a supporting role in a motion picture',
    'best performance by an actor in a supporting role in a motion picture',
    'best director - motion picture', 'best screenplay - motion picture',
    'best original score - motion picture',
    'best original song - motion picture', 'best television series - drama',
    'best performance by an actress in a television series - drama',
    'best performance by an actor in a television series - drama',
    'best television series - comedy or musical',
    'best performance by an actress in a television series - comedy or musical',
    'best performance by an actor in a television series - comedy or musical',
    'best mini-series or motion picture made for television',
    'best performance by an actress in a mini-series or motion picture made for television',
    'best performance by an actor in a mini-series or motion picture made for television',
    'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television',
    'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television'
]


def get_hosts(year):
    """Hosts is a list of one or more strings. Do NOT change the name of this function or what it returns."""
    print('Running Hosts...')
    hosts = run_hosts(year)

    with open(f"{year}Hosts.txt", "w", encoding="utf-8") as f:
        f.write(f"Hosts: {hosts}")
    return hosts


def get_awards(year):
    """Awards is a list of strings. Do NOT change the name of this function or what it returns."""
    print('Running Awards...')
    awards = run_awards(year)
    with open(f"{year}Awards.txt", "w", encoding="utf-8") as f:
        f.write("Awards:\n")
        for a in awards:
            f.write(f"{a},\n")
    return awards


def get_nominees(year):
    """Nominees is a dict with the hard-coded award names as keys, each value a list of strings."""
    print('Running Nominees...')
    start = time.time()
    nominees = run_nominees(year)
    with open(f"{year}Nominees.txt", "w", encoding="utf-8") as f:
        for key in nominees:
            f.write(f"Category: {key}\nNominees: {nominees[key]} \n")
    print('Nominees Time:', time.time() - start)
    return nominees


def get_winner(year):
    """Winners is a dict with the hard-coded award names as keys, each value a single string."""
    print('Running Winners...')
    start = time.time()
    winners = run_winners(year)
    with open(f"{year}Winners.txt", "w", encoding="utf-8") as f:
        for key in winners:
            f.write(f"Category: {key}\nWinner: {winners[key]} \n")
    print('Winners Time:', time.time() - start)
    return winners


def get_presenters(year):
    """Presenters is a dict with the hard-coded award names as keys, each value a list of strings."""
    print('Running Presenters...')
    start = time.time()
    presenters = run_presenters(year)
    with open(f"{year}Presenters.txt", "w", encoding="utf-8") as f:
        for key in presenters:
            f.write(f"Category: {key}\nPresenter: {presenters[key]} \n")
    print('Presenters Time:', time.time() - start)
    return presenters


def pre_ceremony():
    """Load/fetch/process any data the program will use. First thing the TA will run."""
    print("Running pre-ceremony checks (NLTK data, spaCy model)...")
    _ensure_nltk()
    _ensure_spacy_model()
    print("Pre-ceremony processing complete.")
    return


def _parse_args(argv):
    # Default years and parts (mirrors autograder defaults)
    years = ["2013"]
    parts = ["hosts", "awards", "nominees", "presenters", "winner"]

    # Filter years if the user provided one
    provided_years = [a for a in argv if a in {"2013", "2015"}]
    if provided_years:
        years = provided_years

    provided_parts = [a for a in argv if a in parts]
    if provided_parts:
        parts = provided_parts

    return years, parts


def main():
    """Run the pipeline from the CLI. This is the second thing the TA will run."""
    # 1) Always run pre-ceremony once
    pre_ceremony()
    # 2) Parse what to run
    years, parts = _parse_args(sys.argv[1:])
    print(f"Years: {years}")
    print(f"Parts: {parts}")

    # 3) Run each requested part for each requested year
    for y in years:
        print(f"\n=== YEAR {y} ===")
        outputs = {}
        if "hosts" in parts:
            outputs["hosts"] = get_hosts(y)
        if "awards" in parts:
            outputs["awards"] = get_awards(y)
        if "nominees" in parts:
            outputs["nominees"] = get_nominees(y)
        if "presenters" in parts:
            outputs["presenters"] = get_presenters(y)
        if "winner" in parts:
            outputs["winner"] = get_winner(y)

        # Write a combined JSON summary as a convenience
        try:
            with open(f"gg{y}_outputs.json", "w", encoding="utf-8") as jf:
                json.dump(outputs, jf, indent=2, ensure_ascii=False)
            print(f"Wrote gg{y}_outputs.json")
        except Exception as e:
            print(f"(Skipping JSON write: {e})")


if __name__ == '__main__':
    main()