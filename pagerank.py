import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    """Return a probability distribution over which page to visit next."""
    next_page_probs = {}
    num_pages = len(corpus)

    for p in corpus:
        # Probability of jumping to a random page
        next_page_probs[p] = (1 - damping_factor) / num_pages

    if page in corpus:
        linked_pages = corpus[page]
        num_links = len(linked_pages)

        # Probability of following a link from the current page
        for p in corpus:
            if p in linked_pages:
                next_page_probs[p] += damping_factor / num_links
            else:
                # Handle pages with no outgoing links
                if num_links == 0:
                    next_page_probs[p] += damping_factor / num_pages

    return next_page_probs


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {}
    for page in corpus:
        ranks[page] = 0

    # Initialize with a random page
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        ranks[page] += 1
        next_page_probs = transition_model(corpus, page, damping_factor)
        page = random.choices(list(corpus.keys()),
                              weights=list(next_page_probs.values()))[0]

    for page in ranks:
        ranks[page] /= n

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    ranks = {}
    for page in corpus:
        ranks[page] = 1 / num_pages

    while True:
        new_ranks = {}
        for page in corpus:
            new_rank = (1 - damping_factor) / num_pages
            for p in corpus:
                if page in corpus[p]:
                    new_rank += damping_factor * ranks[p] / len(corpus[p])
            new_ranks[page] = new_rank

        max_diff = 0
        for page in corpus:
            max_diff = max(max_diff, abs(ranks[page] - new_ranks[page]))
        ranks = new_ranks

        if max_diff < 0.001:
            break

    return ranks


if __name__ == "__main__":
    main()
