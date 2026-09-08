"""Build the W5C1 recipe corpus: 100 recipes, indexed by their ingredients.

Hand-written, so it is reproducible by construction. The vocabulary is tuned so
that no recipe ever contains a whole pantry: the exercise is ranking partial
matches, not finding a string. Only the ingredients are indexed, never the name.
See instructor-notes.md for the design and the measured numbers.
"""
import csv
from pathlib import Path

RECIPES = [
    # -- middle eastern / mediterranean --------------------------------------
    ("Hummus", "chickpeas, tahini, lemon, garlic, olive oil, cumin, salt"),
    ("Chickpea and parsley salad", "chickpeas, parsley, lemon, red onion, olive oil, salt, pepper"),
    ("Falafel", "chickpeas, parsley, coriander, cumin, garlic, onion, flour, salt"),
    ("Baba ganoush", "aubergine, tahini, lemon, garlic, olive oil, salt"),
    ("Fattoush", "stale bread, cucumber, tomato, sumac, mint, parsley, lemon, olive oil, salt"),
    ("Tabbouleh", "bulgur, parsley, mint, tomato, lemon, olive oil, salt"),
    ("Muhammara", "red pepper, walnuts, pomegranate molasses, breadcrumbs, garlic, olive oil, salt"),
    ("Shakshuka", "eggs, tomato, red pepper, onion, garlic, cumin, paprika, olive oil, salt"),
    ("Lentil soup with lemon", "red lentils, onion, garlic, cumin, lemon, olive oil, salt"),
    ("Roast aubergine with yoghurt", "aubergine, yoghurt, garlic, lemon, olive oil, mint, salt"),
    # -- indian ---------------------------------------------------------------
    ("Chana masala", "chickpeas, tomato, onion, garlic, ginger, cumin, coriander, chilli, salt"),
    ("Chana saag", "chickpeas, spinach, yoghurt, cumin, garlic, ginger, onion, salt"),
    ("Saag paneer", "spinach, paneer, yoghurt, garlic, ginger, garam masala, butter, salt"),
    ("Dal tarka", "red lentils, onion, garlic, ginger, cumin, turmeric, butter, salt"),
    ("Aloo gobi", "potato, cauliflower, onion, cumin, turmeric, ginger, coriander, salt"),
    ("Chicken korma", "chicken, yoghurt, almonds, onion, garlic, ginger, cardamom, cream, salt"),
    ("Rajma", "kidney beans, tomato, onion, garlic, ginger, garam masala, salt"),
    ("Jeera rice", "rice, cumin, butter, salt"),
    ("Raita", "yoghurt, cucumber, mint, cumin, salt"),
    ("Bombay potatoes", "potato, mustard seeds, turmeric, chilli, coriander, onion, salt"),
    # -- italian --------------------------------------------------------------
    ("Pasta e ceci", "chickpeas, pasta, rosemary, garlic, tomato, olive oil, salt"),
    ("Cacio e pepe", "pasta, pecorino, pepper, salt"),
    ("Carbonara", "pasta, eggs, pecorino, guanciale, pepper, salt"),
    ("Puttanesca", "pasta, tomato, anchovy, capers, olives, garlic, chilli, olive oil, salt"),
    ("Aglio e olio", "pasta, garlic, chilli, olive oil, parsley, salt"),
    ("Panzanella", "stale bread, tomato, cucumber, red onion, basil, olive oil, vinegar, salt"),
    ("Ribollita", "stale bread, cannellini beans, kale, carrot, celery, onion, olive oil, salt"),
    ("Risotto alla milanese", "rice, saffron, onion, butter, parmesan, chicken stock, white wine, salt"),
    ("Mushroom risotto", "rice, mushrooms, onion, butter, parmesan, chicken stock, white wine, salt"),
    ("Minestrone", "cannellini beans, carrot, celery, onion, tomato, pasta, olive oil, salt"),
    ("Caponata", "aubergine, celery, tomato, capers, olives, vinegar, sugar, olive oil, salt"),
    ("Pesto", "basil, pine nuts, garlic, parmesan, olive oil, salt"),
    ("Tiramisu", "mascarpone, eggs, sugar, coffee, cocoa, sponge fingers"),
    ("Panna cotta", "cream, sugar, vanilla, gelatine"),
    ("Focaccia", "flour, yeast, olive oil, salt, rosemary, water"),
    # -- spanish / portuguese -------------------------------------------------
    ("Paella", "rice, saffron, chicken, chorizo, peas, red pepper, onion, garlic, chicken stock, salt"),
    ("Gazpacho", "tomato, cucumber, red pepper, garlic, stale bread, olive oil, vinegar, salt"),
    ("Tortilla espanola", "potato, eggs, onion, olive oil, salt"),
    ("Patatas bravas", "potato, tomato, paprika, garlic, olive oil, salt"),
    ("Chorizo and chickpea stew", "chorizo, chickpeas, tomato, onion, garlic, paprika, salt"),
    ("Garlic prawns", "prawns, garlic, chilli, olive oil, parsley, salt"),
    ("Caldo verde", "potato, kale, chorizo, onion, garlic, olive oil, salt"),
    # -- french ---------------------------------------------------------------
    ("French onion soup", "onion, butter, beef stock, stale bread, gruyere, thyme, salt"),
    ("Ratatouille", "aubergine, courgette, tomato, red pepper, onion, garlic, thyme, olive oil, salt"),
    ("Coq au vin", "chicken, red wine, bacon, mushrooms, onion, garlic, thyme, butter, salt"),
    ("Quiche lorraine", "flour, butter, eggs, cream, bacon, gruyere, salt"),
    ("Croque monsieur", "bread, ham, gruyere, butter, flour, milk, mustard, salt"),
    ("Soupe au pistou", "cannellini beans, courgette, carrot, basil, garlic, parmesan, olive oil, salt"),
    ("Crepes", "flour, eggs, milk, butter, sugar, salt"),
    ("Creme brulee", "cream, eggs, sugar, vanilla"),
    # -- east and southeast asian ---------------------------------------------
    ("Miso soup", "miso, dashi, tofu, spring onion, seaweed"),
    ("Miso ramen", "miso, noodles, pork, spring onion, egg, garlic, ginger, chicken stock"),
    ("Mapo tofu", "tofu, pork, doubanjiang, sichuan pepper, spring onion, garlic, ginger, soy sauce"),
    ("Fried rice", "rice, egg, spring onion, peas, soy sauce, garlic, sesame oil"),
    ("Kimchi jjigae", "kimchi, pork, tofu, spring onion, gochujang, garlic, soy sauce"),
    ("Bibimbap", "rice, spinach, carrot, egg, gochujang, sesame oil, soy sauce, garlic"),
    ("Pad thai", "noodles, tamarind, fish sauce, egg, peanuts, bean sprouts, spring onion, sugar"),
    ("Green curry", "coconut milk, green curry paste, chicken, aubergine, basil, fish sauce, sugar"),
    ("Tom yum", "prawns, lemongrass, lime, chilli, fish sauce, mushrooms, coriander"),
    ("Teriyaki salmon", "salmon, soy sauce, mirin, sugar, ginger, sesame seeds"),
    ("Gyoza", "pork, cabbage, garlic, ginger, spring onion, soy sauce, flour"),
    ("Chicken katsu", "chicken, breadcrumbs, egg, flour, soy sauce, sugar, onion, carrot"),
    # -- british / american ---------------------------------------------------
    ("Shepherd's pie", "lamb mince, potato, carrot, onion, tomato, beef stock, butter, salt"),
    ("Cottage pie", "beef mince, potato, carrot, onion, beef stock, butter, salt"),
    ("Fish pie", "white fish, potato, milk, butter, flour, parsley, egg, salt"),
    ("Toad in the hole", "sausages, flour, eggs, milk, salt"),
    ("Bubble and squeak", "potato, cabbage, onion, butter, salt"),
    ("Bread and butter pudding", "stale bread, butter, eggs, milk, sugar, raisins, nutmeg"),
    ("Scones", "flour, butter, sugar, milk, baking powder, salt"),
    ("Victoria sponge", "flour, butter, sugar, eggs, baking powder, jam, vanilla"),
    ("Shortbread", "flour, butter, sugar, salt"),
    ("Apple crumble", "apples, flour, butter, sugar, oats, cinnamon"),
    ("Banana bread", "bananas, flour, butter, sugar, eggs, baking powder, cinnamon"),
    ("Pancakes", "flour, eggs, milk, butter, sugar, baking powder, salt"),
    ("Cornbread", "cornmeal, flour, milk, eggs, butter, sugar, baking powder, salt"),
    ("Mac and cheese", "pasta, cheddar, milk, butter, flour, mustard, breadcrumbs, salt"),
    ("Clam chowder", "clams, potato, bacon, onion, cream, celery, thyme, salt"),
    ("Chilli con carne", "beef mince, kidney beans, tomato, onion, garlic, cumin, chilli, paprika, salt"),
    ("Cornish pasty", "flour, butter, beef, potato, swede, onion, pepper, salt"),
    ("Fish and chips", "white fish, potato, flour, beer, salt"),
    # -- mexican / latin ------------------------------------------------------
    ("Guacamole", "avocado, lime, coriander, red onion, chilli, salt"),
    ("Salsa roja", "tomato, chilli, onion, garlic, coriander, lime, salt"),
    ("Refried beans", "pinto beans, onion, garlic, cumin, lard, salt"),
    ("Tacos al pastor", "pork, pineapple, achiote, chilli, onion, coriander, lime"),
    ("Mole poblano", "chilli, chocolate, tomato, almonds, sesame seeds, cinnamon, chicken stock"),
    ("Ceviche", "white fish, lime, red onion, coriander, chilli, salt"),
    ("Black bean soup", "black beans, onion, garlic, cumin, coriander, lime, chicken stock, salt"),
    ("Arroz con pollo", "rice, chicken, onion, garlic, red pepper, peas, chicken stock, saffron, salt"),
    # -- salads and sides -----------------------------------------------------
    ("Greek salad", "tomato, cucumber, feta, olives, red onion, oregano, olive oil, salt"),
    ("Caesar salad", "lettuce, anchovy, parmesan, egg, garlic, lemon, stale bread, olive oil"),
    ("Waldorf salad", "apples, celery, walnuts, grapes, mayonnaise, lemon, salt"),
    ("Coleslaw", "cabbage, carrot, mayonnaise, vinegar, mustard, salt"),
    ("Potato salad", "potato, mayonnaise, mustard, spring onion, egg, vinegar, salt"),
    ("Tzatziki", "yoghurt, cucumber, garlic, mint, lemon, olive oil, salt"),
    ("Beetroot and feta salad", "beetroot, feta, walnuts, rocket, lemon, olive oil, salt"),
    ("Roasted carrot and harissa salad", "carrot, harissa, yoghurt, coriander, lemon, olive oil, salt"),
    ("Cauliflower with zaatar", "cauliflower, zaatar, lemon, olive oil, garlic, salt"),
    ("Braised lentils with anchovy", "lentils, anchovy, carrot, celery, onion, garlic, olive oil, salt"),
    ("Nduja pasta", "pasta, nduja, tomato, garlic, parmesan, olive oil, salt"),
    ("Spiced chickpeas with sumac", "chickpeas, sumac, red onion, parsley, lemon, olive oil, salt"),
]


def main() -> None:
    assert len(RECIPES) == 100, f"expected 100 recipes, got {len(RECIPES)}"
    out = Path(__file__).resolve().parents[1] / "exercise" / "data" / "recipes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["recipe_id", "name", "ingredients"])
        w.writeheader()
        for i, (name, ingredients) in enumerate(RECIPES, 1):
            w.writerow({"recipe_id": f"r{i:03d}", "name": name,
                        "ingredients": ingredients})
    print(f"wrote {out} with {len(RECIPES)} recipes")


if __name__ == "__main__":
    main()
