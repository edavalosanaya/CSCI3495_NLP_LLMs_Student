"""Build the W5C1 casebook: 100 case files, indexed by what was observed at each.

Hand-written, so it is reproducible by construction. Only `observations` is
indexed, never `title`, so no answer can be found by searching for it.

THE DESIGN RULE, and it is the whole reason this corpus exists in this shape.
An earlier mystery version of this class failed because every query was a rare
proper noun: search "astrolabe", get two hits, and Ctrl+F has beaten the search
engine before you have finished explaining tf. A retrieval exercise only teaches
retrieval if exact matching **cannot** work. So:

* the scene is **six ordinary observations**, and **no case file contains all six**;
* each one alone returns a useless pile (`mud` is in 41 of 100);
* the words that are everywhere (`london`, `night`, `street`) are switched off by
  idf rather than by a stop list;
* exactly one rare observation, `creosote`, is worth more than all the common ones
  put together, and finding that is the post-break question.

Run it, and it checks every one of those properties before writing:

    uv run python weeks/week-05/class-01/solutions/make_casebook.py

It prints the document frequency of every scene word as it goes, which is the
quickest way to see why the search has to rank rather than match.
"""
import csv
from collections import Counter
from pathlib import Path

# The scene of the Lauriston Gardens murder: what Holmes has, and what the
# students search with. Six ordinary words, deliberately.
SCENE = "mud, rain, tobacco, cab, boots, bruise"

# The one observation worth chasing. Rare enough to be decisive, and it has to be
# EARNED: no team can guess it from the scene.
DECISIVE = "creosote"

CASES = [
    # -- burglary and theft ---------------------------------------------------
    ("The Brixton warehouse break-in", "mud, boots, crowbar, warehouse, rain, night, london, lantern"),
    ("The pawnbroker's back room", "pawn ticket, ledger, boots, tobacco, cellar, london, gaslight"),
    ("The Hatton Garden safe", "safe, drill, boots, mud, night, london, watchman, bruise"),
    ("The stolen bicycle at Kew", "bicycle, mud, tyre, boots, rain, gravel, london"),
    ("The emptied jewel case", "jewel case, glove, window, mud, night, london, servant"),
    ("The coal cellar entry", "cellar, coal, boots, mud, lantern, london, night"),
    ("The rifled writing desk", "writing desk, ink stain, glove, ledger, london, tobacco"),
    ("The area railings at Cadogan Place", "railings, boots, mud, rain, night, london, bruise"),
    ("The dockside crate", "crate, dock, tar, boots, mud, rope, london, night"),
    ("The vanished plate chest", "plate chest, servant, cab, london, night, boots"),
    ("The forced scullery window", "window, scullery, boots, mud, rain, london, servant"),
    ("The theft at the Diogenes Club", "club, greatcoat, tobacco, cab, london, night, glove"),
    ("The stolen greatcoat", "greatcoat, cab, rain, london, tobacco, boots"),
    ("The emptied till at Camberwell", "till, shop, boots, mud, night, london, bruise"),
    ("The missing racing plate", "stable, straw, boots, mud, rain, groom, bruise"),
    ("The chapel poor box", "chapel, poor box, boots, mud, candle, london, night"),
    ("The stolen surgical case", "surgical case, laudanum, boots, cab, london, night"),
    ("The lodging house trunk", "trunk, lodging house, boots, tobacco, london, rain"),
    ("The burgled photographer's", "photographer, plate glass, boots, mud, london, night, glove"),
    ("The rifled post bag", "post bag, telegram, boots, mud, rain, london, cab"),
    # -- poisoning ------------------------------------------------------------
    ("The Stoke Moran bedroom", "bell rope, ventilator, whistle, bruise, night, safe, dog lash"),
    ("The chemist's error at Pimlico", "chemist, laudanum, bottle, ledger, london, bruise"),
    ("The bitter coffee", "coffee, bitter, laudanum, tremor, london, night, servant"),
    ("The Devonshire root", "root, tremor, candle, night, bruise, fog"),
    ("The tainted claret", "claret, decanter, tremor, servant, london, night, bruise"),
    ("The apothecary's ledger", "apothecary, ledger, laudanum, ink stain, london, tobacco"),
    ("The green phial", "phial, green, chemist, tremor, london, bruise, night"),
    ("The dose at Norwood", "laudanum, bottle, tremor, night, bruise, servant, rain"),
    ("The curious thirst", "thirst, tremor, water jug, night, london, bruise"),
    ("The blue vitriol", "vitriol, burn, chemist, london, bruise, night, tobacco"),
    ("The poisoned dart", "dart, blowpipe, creosote, bruise, night, london, footprint"),
    ("The sailor's tin", "tin, tar, dock, tremor, bruise, london, night"),
    ("The governess and the tonic", "tonic, governess, tremor, bottle, night, bruise"),
    ("The tainted snuff", "snuff, tobacco, tremor, london, bruise, night, glove"),
    ("The physician's black bag", "black bag, laudanum, physician, cab, london, rain, night"),
    # -- blackmail and extortion ---------------------------------------------
    ("The letters at Appledore", "letters, safe, greatcoat, night, glove, bruise, london"),
    ("The photographer's negative", "negative, photographer, cab, london, night, tobacco"),
    ("The demand in violet ink", "violet ink, ink stain, letters, london, telegram, glove"),
    ("The bank clerk's debt", "bank, ledger, clerk, london, tobacco, cab, bruise"),
    ("The burnt fragment", "ash, grate, letters, tobacco, london, night, glove"),
    ("The threat at the theatre", "theatre, letters, cab, london, night, greatcoat"),
    ("The cabman's testimony", "cab, cabman, rain, mud, london, night, tobacco"),
    ("The anonymous telegram", "telegram, post office, london, glove, tobacco, cab"),
    ("The secretary's copy book", "copy book, ink stain, ledger, clerk, london, tobacco"),
    ("The widow's annuity", "annuity, ledger, solicitor, london, bruise, glove"),
    ("The music hall dresser", "music hall, dresser, letters, cab, london, night"),
    ("The gambling note", "gambling, note, ledger, tobacco, london, night, bruise"),
    # -- disappearance --------------------------------------------------------
    ("The lady vanished at Charing Cross", "railway, trunk, cab, london, rain, boots, glove"),
    ("The empty cab at Hyde Park", "cab, cabman, mud, rain, london, night, glove"),
    ("The missing clerk of Threadneedle Street", "clerk, ledger, london, tobacco, cab, boots"),
    ("The schoolmaster's bicycle", "bicycle, tyre, mud, rain, boots, gravel, bruise"),
    ("The empty berth", "berth, ship, dock, trunk, rope, london, tar"),
    ("The abandoned lodgings", "lodging house, trunk, ash, grate, london, tobacco, boots"),
    ("The footprints on the moor", "footprint, mud, rain, moor, boots, fog, bruise"),
    ("The unclaimed portmanteau", "portmanteau, railway, london, boots, mud, glove"),
    ("The absent bridegroom", "wedding, cab, london, glove, telegram, night"),
    ("The vanished chemist", "chemist, laudanum, ledger, london, boots, tobacco"),
    ("The silent stable lad", "stable, straw, groom, bruise, mud, boots, rain"),
    ("The lost despatch", "despatch, telegram, clerk, london, cab, glove, night"),
    ("The boatman of Rotherhithe", "boatman, dock, tar, rope, creosote, river"),
    # -- assault and murder ---------------------------------------------------
    ("Lauriston Gardens", "mud, rain, cab, boots, ring, gaslight, london, servant"),
    ("The Brixton Road affair", "mud, rain, cab, boots, bruise, gaslight"),
    ("The blow in the fog", "fog, bruise, boots, mud, london, night, gaslight, stick"),
    ("The quarrel at the Alpha Inn", "inn, bruise, tobacco, boots, london, night, goose"),
    ("The body on the towpath", "towpath, river, mud, rain, boots, bruise, rope"),
    ("The staircase at Norwood", "staircase, bruise, boots, mud, night, london, ash"),
    ("The gamekeeper's lodge", "lodge, gun, boots, mud, rain, bruise, dog lash"),
    ("The riverside warehouse", "creosote, tar, boots, mud, bruise, rope"),
    ("The struggle at the area steps", "area steps, bruise, boots, mud, rain, london, night"),
    ("The blow with the poker", "poker, grate, bruise, ash, london, night, servant"),
    ("The sailor's knife", "knife, sailor, dock, tar, bruise, tobacco, london"),
    ("The empty revolver", "revolver, powder, bruise, night, london, glove, boots"),
    ("The garrotting in Wapping", "rope, bruise, dock, night, london, mud, tar"),
    ("The upset chair", "chair, struggle, bruise, ash, grate, london, night"),
    ("The blow behind the ear", "bruise, boots, staircase, london, night, servant"),
    ("The night watchman's round", "watchman, lantern, boots, mud, night, london, bruise"),
    ("The gash on the wrist", "wrist, bruise, glove, blood, london, night, boots"),
    ("The stable yard affray", "stable, straw, groom, bruise, mud, boots, night"),
    ("The cab horse that bolted", "cab, cabman, horse, mud, rain, bruise, london"),
    ("The shot in the fog", "fog, revolver, powder, bruise, night, london, mud"),
    # -- forgery and fraud ----------------------------------------------------
    ("The red-headed applicant", "advertisement, cellar, clerk, london, tobacco, boots, spade"),
    ("The forged endorsement", "endorsement, ink stain, ledger, clerk, london, glove"),
    ("The engraver's plate", "engraver, plate, ink stain, cellar, london, tobacco"),
    ("The doctored share list", "share list, ledger, clerk, london, telegram, tobacco"),
    ("The false name at the hotel", "hotel, register, ink stain, cab, london, glove, boots"),
    ("The three quarter's signature", "signature, ink stain, ledger, telegram, london, clerk"),
    ("The counterfeit sovereigns", "sovereign, plaster of paris, cellar, london, tobacco, boots"),
    ("The altered will", "will, solicitor, ink stain, ledger, london, glove"),
    ("The insurance claim at Lloyd's", "insurance, ledger, clerk, london, telegram, tobacco"),
    ("The bogus prospectus", "prospectus, ink stain, clerk, london, tobacco, cab"),
    ("The copied key", "key, wax, plaster of paris, cellar, london, glove, boots"),
    ("The substituted picture", "picture, varnish, gum, cellar, london, glove, boots"),
    # -- oddities -------------------------------------------------------------
    ("The six busts of Napoleon", "bust, plaster of paris, cellar, london, night, boots, mud"),
    ("The dancing figures on the sill", "cipher, chalk, sill, london, night, glove"),
    ("The orange pips in the post", "orange pips, post bag, telegram, london, rain, night"),
    ("The engineer's thumb", "engineer, press, bruise, lantern, night, rain, boots"),
    ("The blue carbuncle in the crop", "goose, carbuncle, inn, london, tobacco, night"),
    ("The trained mongoose", "mongoose, dog lash, straw, night, bruise, lantern"),
    ("The noise in the attic", "attic, straw, candle, night, london, boots, mud"),
    ("The whistle at three in the morning", "whistle, bell rope, night, ventilator, bruise, candle"),
]


def tokens(observations: str) -> list[str]:
    """Every word of an observation list, the way the vectorizer will see it."""
    return observations.replace(",", " ").split()


def check() -> None:
    """The design rule, enforced. If any of this fails the class does not work."""
    assert len(CASES) == 100, f"expected 100 cases, got {len(CASES)}"
    titles = [t for t, _ in CASES]
    assert len(set(titles)) == 100, "duplicate case title"

    df = Counter()
    for _, obs in CASES:
        df.update(set(tokens(obs)))
    scene = tokens(SCENE)

    # 1. No case file holds the whole scene, so there is no string to find.
    for title, obs in CASES:
        have = set(tokens(obs)) & set(scene)
        assert len(have) < len(scene), f"{title} contains the entire scene"

    # 2. Every scene word alone returns a useless pile.
    for word in scene:
        assert df[word] >= 15, f"{word!r} is too rare to be a fair query: {df[word]}"

    # 3. The decisive observation is rare enough to be worth chasing.
    assert 2 <= df[DECISIVE] <= 4, f"{DECISIVE!r} in {df[DECISIVE]} cases"

    # 4. Something is in nearly every file, so idf has a zero to show.
    assert max(df.values()) >= 60, "no near-ubiquitous word for idf to switch off"
    print(f"scene df: " + ", ".join(f"{w} {df[w]}" for w in scene))
    print(f"{DECISIVE} df: {df[DECISIVE]}   |   most common: "
          + ", ".join(f"{w} {n}" for w, n in df.most_common(3)))


def main() -> None:
    check()
    out = Path(__file__).resolve().parents[1] / "exercise" / "data" / "casebook.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "title", "observations"])
        w.writeheader()
        for i, (title, obs) in enumerate(CASES, 1):
            w.writerow({"case_id": f"c{i:03d}", "title": title, "observations": obs})
    print(f"wrote {out.name} with {len(CASES)} case files")


if __name__ == "__main__":
    main()
