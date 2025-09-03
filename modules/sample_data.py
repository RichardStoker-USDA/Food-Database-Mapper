"""
Sample datasets with 25 items each showing varying similarity scores
"""

# 25 input food descriptions with varying complexity
sample_input_data = [
    # Excellent matches (should score 0.90-0.94)
    ["1", "apple juice"],
    ["2", "chicken breast grilled"],
    ["3", "whole milk"],
    
    # Very good matches (should score 0.88-0.92)
    ["4", "orange juice fresh"],
    ["5", "bread whole wheat"],
    ["6", "cheddar cheese"],
    ["7", "scrambled eggs"],
    
    # Good matches (should score 0.85-0.90)
    ["8", "pasta with tomato sauce"],
    ["9", "beef steak medium rare"],
    ["10", "yogurt plain"],
    ["11", "brown rice cooked"],
    ["12", "salmon fillet baked"],
    
    # Moderate matches (should score 0.82-0.87)
    ["13", "vegetable soup"],
    ["14", "fruit salad mixed"],
    ["15", "pizza slice pepperoni"],
    ["16", "ice cream vanilla"],
    ["17", "coffee with cream"],
    
    # Weaker matches (should score 0.80-0.85)
    ["18", "energy drink"],
    ["19", "protein bar chocolate"],
    ["20", "trail mix nuts"],
    ["21", "smoothie berry"],
    
    # NO MATCH items (should score < 0.85)
    ["22", "xyz123 test item"],
    ["23", "random text here"],
    ["24", "unknown food item 999"],
    ["25", "synthetic compound ABC"]
]

# 25 target reference descriptions
sample_target_data = [
    ["A001", "Apple juice, unsweetened, bottled, without added ascorbic acid"],
    ["A002", "Chicken, broilers or fryers, breast, meat only, cooked, grilled"],
    ["A003", "Milk, whole, 3.25% milkfat, with added vitamin D"],
    ["A004", "Orange juice, raw, includes from concentrate, fortified with calcium"],
    ["A005", "Bread, whole-wheat, commercially prepared, toasted"],
    ["A006", "Cheese, cheddar, sharp, sliced"],
    ["A007", "Egg, whole, cooked, scrambled, with added fat"],
    ["A008", "Pasta, cooked, enriched, with added salt"],
    ["A009", "Beef, loin, tenderloin steak, boneless, separable lean only, trimmed to 0 fat, select, cooked, grilled"],
    ["A010", "Yogurt, plain, whole milk, 8 grams protein per 8 ounce"],
    ["A011", "Rice, brown, long-grain, cooked, enriched"],
    ["A012", "Fish, salmon, Atlantic, farmed, cooked, dry heat"],
    ["A013", "Soup, vegetable beef, canned, condensed, single brand"],
    ["A014", "Salad, fruit, fresh, without dressing"],
    ["A015", "Pizza, meat topping, thick crust, frozen, cooked"],
    ["A016", "Ice creams, vanilla, rich, 16% fat"],
    ["A017", "Coffee, brewed from grounds, prepared with tap water, decaffeinated"],
    ["A018", "Beverages, Energy drink, RED BULL, sugar free, with added caffeine, niacin, pantothenic acid, vitamins B6 and B12"],
    ["A019", "Snacks, granola bar, chocolate chip, hard"],
    ["A020", "Nuts, mixed nuts, dry roasted, with peanuts, without salt added"],
    ["A021", "Beverages, fruit juice drink, reduced sugar, with vitamin E added"],
    ["A022", "Potatoes, baked, flesh and skin, with salt"],
    ["A023", "Carrots, raw, baby, organic"],
    ["A024", "Tomatoes, red, ripe, canned, packed in tomato juice, no salt added"],
    ["A025", "Broccoli, cooked, boiled, drained, without salt"]
]

def get_sample_input_csv():
    """Returns sample input data as CSV string"""
    header = "id,food_description\n"
    rows = [f"{row[0]},{row[1]}" for row in sample_input_data]
    return header + "\n".join(rows)

def get_sample_target_csv():
    """Returns sample target data as CSV string"""
    header = "code,food_name\n"
    rows = [f'{row[0]},"{row[1]}"' for row in sample_target_data]
    return header + "\n".join(rows)