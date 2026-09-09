"""Five small domain corpora for the tokenizer bake-off.

Each entry has a `train` string, which the team's BPE learns its merges from,
and a `test` string, which every team's tokenizer is scored on. They are short
on purpose: 300 merges over a few hundred words runs in under a second.
"""

CORPORA = {
    "english": {
        "train": """
        The morning train was late again, so she walked the last mile along the
        river and watched the water move. It was the kind of ordinary walk that
        turns into a habit without anyone deciding on it. By the end of the month
        she knew which bench was dry after rain, which gate stuck, and which dog
        would bark at the gulls. The city kept doing what cities do, and the
        river kept not caring. She thought about the letter she had not answered
        and decided that tomorrow would be soon enough. There were three ducks
        under the bridge and a man selling coffee out of a cart, and the coffee
        was bad but the morning was good, and she walked on to work with her
        hands in her pockets thinking about nothing much at all, which was the
        whole point of walking, and the point of the river, and the reason she
        would be late again tomorrow and the day after that as well.
        """,
        "test": """
        She walked to the river again and watched the water move under the
        bridge, thinking about the letter and about nothing much at all.
        """,
    },
    "python": {
        "train": """
        def load_dataset(path, limit=None):
            rows = []
            with open(path, encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if limit is not None and i >= limit:
                        break
                    rows.append(json.loads(line))
            return rows

        class Counter(dict):
            def add(self, key, value=1):
                self[key] = self.get(key, 0) + value
                return self

            def top(self, n=10):
                return sorted(self.items(), key=lambda kv: -kv[1])[:n]

        def train_test_split(rows, fraction=0.8, seed=0):
            random.Random(seed).shuffle(rows)
            cut = int(len(rows) * fraction)
            return rows[:cut], rows[cut:]

        if __name__ == "__main__":
            rows = load_dataset("reviews.jsonl", limit=5000)
            train, test = train_test_split(rows)
            print(f"train={len(train)} test={len(test)}")
        """,
        "test": """
        def score(model, rows):
            correct = sum(1 for row in rows if model.predict(row["text"]) == row["label"])
            return correct / len(rows)
        """,
    },
    "spanish": {
        "train": """
        El tren de la manana llegaba tarde otra vez, asi que ella caminaba el
        ultimo kilometro junto al rio y miraba moverse el agua. Era la clase de
        paseo corriente que se convierte en costumbre sin que nadie lo decida.
        Al final del mes ya sabia que banco quedaba seco despues de la lluvia,
        que puerta se atascaba y que perro ladraba a las gaviotas. La ciudad
        seguia haciendo lo que hacen las ciudades, y al rio no le importaba
        nada. Pensaba en la carta que no habia contestado y decidia que manana
        seria bastante pronto. Habia tres patos debajo del puente y un hombre
        que vendia cafe en un carrito, y el cafe era malo pero la manana era
        buena, y ella seguia caminando hacia el trabajo con las manos en los
        bolsillos pensando en casi nada, que era justamente el sentido de
        caminar, y el sentido del rio, y la razon por la que volveria a llegar
        tarde manana y tambien pasado manana.
        """,
        "test": """
        Ella caminaba junto al rio otra vez y miraba moverse el agua debajo del
        puente, pensando en la carta y en casi nada.
        """,
    },
    "chat": {
        "train": """
        omg no way lol did that actually happen
        yeah fr fr i was there 😭😭 it was so bad
        wait what happened tho
        ok so basically we got there like 20 min late and the door was locked lmao
        LMAOOO no bc why is that always us
        idk man idk 💀 anyway u free tmrw
        prob yeah what time
        idk like 7ish? we can figure it out later
        bet 🔥
        omw btw bring the charger u borrowed pls
        oh shoot i forgot it srry 😭 ill bring it fr this time
        u said that last time lol
        ik ik i know i know but this time fr
        ok whatever lol see u tmrw
        wait one more thing did u see the thing
        what thing
        THE THING
        omg no send it
        sending 💀💀💀
        """,
        "test": """
        omg wait did u see the thing lol i was there fr 😭 anyway u free tmrw
        like 7ish? bet
        """,
    },
    "chemistry": {
        "train": """
        The oxidation of cyclohexanol to cyclohexanone proceeds cleanly in the
        presence of pyridinium chlorochromate. Hydrogenation of the alkene over
        palladium on carbon gives the saturated hydrocarbon in high yield.
        Acetylation of the amine with acetic anhydride affords the acetamide,
        which crystallizes from ethanol. Treatment of the ester with lithium
        aluminium hydride produces the primary alcohol, and subsequent
        bromination with phosphorus tribromide gives the alkyl bromide.
        Nucleophilic substitution with sodium azide, followed by reduction,
        regenerates the amine. Esterification of benzoic acid with methanol
        under acid catalysis yields methyl benzoate. The nitration of toluene
        gives predominantly the ortho and para isomers, and recrystallization
        from aqueous ethanol separates the para nitrotoluene. Polymerization of
        the methacrylate monomer with a peroxide initiator produces
        poly methyl methacrylate, a transparent thermoplastic.
        """,
        "test": """
        Hydrogenation of the unsaturated ester over palladium on carbon,
        followed by esterification with methanol, gives methyl cyclohexane
        carboxylate.
        """,
    },
}
