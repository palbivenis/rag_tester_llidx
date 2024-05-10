import os
import json
import re
from collections import defaultdict
from config import set_environment
from typing import List
from concurrent import futures
from tqdm import tqdm

# there are some titles that are mistakenly all caps, convert them to title case
special_titles = [
    "APPEALING A DENIED CLAIM",
    "VISION PLANS",
    "MEDICAL SECOND OPINION",
    "HEALTH AND DEPENDENT CARE FLEXIBLE SPENDING ACCOUNTS \(FSAs\)",
    "CONTACT US",
    "DIRECTORY OF PLANS AND ADMINISTRATORS",
    "EXECUTION",
]


def correct_text(documents):
    all_text = [doc.text for doc in documents]
    all_text = " ".join(all_text)
    all_text = all_text.encode("ascii", "ignore").decode()
    # replace all multiple space characters with single space
    all_text = re.sub("[ ]+", " ", all_text)
    # replace extra space before a comma
    all_text = all_text.replace(" ,", ",")
    all_text = all_text.replace(" - ", "$-$")
    all_text = all_text.replace(" -", "-")
    all_text = all_text.replace("$-$", " - ")
    all_text = all_text.replace("HEAL TH", "HEALTH")
    all_text = all_text.replace("MEDICALL Y", "MEDICALLY")
    all_text = all_text.replace(
        "\nMedicare & ACME s Medical Plans    \n",
        "\nMedicare & ACMEs Medical Plans    \n",
    )
    for special_title in special_titles:
        # find all titles in the text
        replacement = special_title.title()
        replacement = replacement.replace("\(", "(").replace("\)", ")")
        all_text = re.sub(f"\n\s*{special_title}\s*\n", f"\n{replacement}\n", all_text)
    return all_text


set_environment()
# eval_directory = os.environ["EVAL_DIRECTORY"]
# reader = SimpleDirectoryReader(eval_directory)
# documents = reader.load_data()
# all_text = correct_text(documents)


headings = json.load(open("./datasets/acme_spd/files/ACME_headings.json", "r"))

ex_titles = {
    "title1": {
        "title1.1": {
            "title1.1.1": {},
            "title1.1.2": {"title1.1.2.1": {}},
        },
    },
    "title2": {"title2.1": {}, "title2.2": {}},
}

ex_text = "Document\ntitle1\ntitle1.1\ntitle2 \ntitle1\ncontents of the title1\ntitle1.1\ncontents of title1.1"


def flatten_titles(table_of_contents, titles={}, parent=None, siblings=None):
    if siblings is None:
        siblings = list(table_of_contents.keys())
    for title, val in table_of_contents.items():
        if val == {}:
            titles[title] = {"parent": parent, "children": {}, "siblings": siblings}
        else:
            titles[title] = {
                "parent": parent,
                "children": list(val.keys()),
                "siblings": siblings,
            }
            titles = flatten_titles(
                val, titles, parent=title, siblings=list(val.keys())
            )
    return titles


def extract_parent_titles(titles, existing_parents=[], parents={}):
    """
    titles is a dictionary of N-levels
    each level is a list of titles
    EX: titles = {
                    'title1': {
                                'title1.1': {
                                    'title1.1.1': [],
                                    'title1.1.2': {
                                        'title1.1.2.1': []
                                    },
                                },
                            },
                    'title2': {
                                'title2.1': [],
                                'title2.2': []
                    }
                    ...
                }
    Goal is to find the parents/grandparents of each node
    return a dictionary where:
    parents = {node_name: [list of its parents]}
    """
    for title, val in titles.items():
        # escape parentheses
        # title = title.replace("(", "\(").replace(")", "\)")
        if val == []:
            parents[title] = existing_parents
        else:
            parents[title] = existing_parents
            parents = extract_parent_titles(val, existing_parents + [title], parents)
    return parents


def get_table_of_contents(titles, prefix=""):
    # Given the titles as a dict of dicts, return the table of contents in 1.1.1 format
    toc = []
    subprefix = 1
    for title, val in titles.items():
        if prefix == "":
            pr = f"{subprefix}"
        else:
            pr = f"{prefix}.{subprefix}"
        toc.append(f"{pr}. {title}")
        if val != {}:
            toc += get_table_of_contents(val, prefix=pr)
        subprefix += 1
    return toc


# chunk the text document based on given title sections
def chunk_text(source_text, table_of_contents):
    """
    titles is a dictionary of N-levels
    each level is a list of titles
    EX: titles = {
                    'title1': {
                                'title1.1': {
                                    'title1.1.1': [],
                                    'title1.1.2': {
                                        'title1.1.2.1': []
                                    },
                                },
                            },
                    'title2': {
                                'title2.1': [],
                                'title2.2': []
                    }
                    ...
                }
    Goal is to fill in the empty lists with the text that falls under that title
    """
    # starts is a dictionary of a mapping from the start character index to the name of the title
    starts_dict = {}
    # first find all titles and subtitles
    flattened_titles = flatten_titles(table_of_contents)
    for title in flattened_titles:
        this_title_starts = [
            m.start() for m in re.finditer(f"\n\s*{re.escape(title)}\s*\n", source_text)
        ]
        if len(this_title_starts) == 0:
            print("Title not found: ", title)
        for title_start in this_title_starts:
            starts_dict[title_start] = title
    # sort the starts by the start location
    starts = sorted(list(starts_dict.keys()))
    title_texts = defaultdict(list)
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(source_text)
        title = starts_dict[start]
        title_texts[title].append(source_text[start:end])
    # Flatten all text for each title
    title_texts = {k: "\n".join(v) for k, v in title_texts.items()}
    # now add the text to the source_text, including all its parents
    for title, text in title_texts.items():
        flattened_titles[title]["text"] = text
        # get all parents/grandparents for this title
        parent = flattened_titles[title]["parent"]
        parents = [parent]
        while parent is not None:
            parents.append(flattened_titles[parent]["parent"])
            parent = flattened_titles[parent]["parent"]
        # remove None from the list
        parents = [p for p in parents if p is not None]
        flattened_titles[title]["parents"] = parents
    return flattened_titles


def threadpool_map(f, args_list: List[dict], num_workers: int = 64, return_exceptions: bool = True) -> list:
    """
    Same as ThreadPoolExecutor.map with option of returning exceptions. Returns results in the same
    order as input `args_list`.
    """
    results = {}
    with tqdm(total=len(args_list)) as progress_bar:
        with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures_dict = {executor.submit(f, **args): ind for ind, args in enumerate(args_list)}
            for future in futures.as_completed(futures_dict):
                ind = futures_dict[future]
                try:
                    results[ind] = future.result()
                except Exception as e:
                    if return_exceptions:
                        results[ind] = e
                    else:
                        raise
                progress_bar.update(1)

    # Reorders the results to be in the same order as the input
    results = [results[ind] for ind in range(len(results))]
    return results
