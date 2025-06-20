# Augmented English Prompts for Drink Picking Task
# Based on the 4 original Chinese prompts and augmentation strategies
import json
import os
from typing import Dict, List

'''
python3 scripts/augmented_english_prompts.py
'''

def generate_categorized_english_prompts():
    """Generate diverse English prompts for drink picking tasks, categorized by drink type"""
    
    # Base drinks mapping
    drinks_mapping = {
        "冰红茶": "red tea",
        "青提绿茶": "green tea", 
        "乳酸菌水乐": "lactic drink",
        "劲凉冰红茶": "cool blue tea"
    }
    
    # Color associations
    color_map = {
        "red tea": "red",
        "green tea": "green", 
        "lactic drink": "pink",
        "cool blue tea": "blue"
    }
    
    # Initialize categorized prompts dictionary
    categorized_prompts = {chinese_name: [] for chinese_name in drinks_mapping.keys()}
    
    # Action variations
    actions = ["Pick", "Grab", "Take", "Get", "Move"]
    
    # Position variations  
    positions = ["top shelf", "bottom shelf", "left shelf", "right shelf", "upper rack", "lower rack"]
    
    # Target variations
    targets = ["box", "container", "basket", "tray"]
    
    # Object descriptions
    object_types = ["bottle", "drink", "beverage", "can"]
    
    # Helper function to categorize prompts by drink type
    def add_prompt_to_category(prompt: str, target_drink: str):
        """Add a prompt to the appropriate drink category"""
        for chinese_name, english_name in drinks_mapping.items():
            if english_name == target_drink:
                categorized_prompts[chinese_name].append(prompt)
                break
    
    def add_prompt_by_color(prompt: str, target_color: str):
        """Add a prompt to the appropriate drink category based on color"""
        for chinese_name, english_name in drinks_mapping.items():
            if color_map.get(english_name) == target_color:
                categorized_prompts[chinese_name].append(prompt)
                break
    
    # 1. Basic action variations
    for chinese_name, english_name in drinks_mapping.items():
        for action in actions:
            categorized_prompts[chinese_name].extend([
                f"{action} {english_name} to box",
                f"{action} the {english_name}",
                f"{action} {english_name} from shelf to box"
            ])
    
    # 2. Color-based descriptions
    for chinese_name, english_name in drinks_mapping.items():
        color = color_map[english_name]
        categorized_prompts[chinese_name].extend([
            f"Pick {color} bottle to box",
            f"Grab {color} drink from shelf",
            f"Take {color} beverage to container",
            f"Get the {color} one",
            f"Pick the {color} bottle",
            f"Grab {color} drink",
            f"Take {color} beverage"
        ])
    
    # 3. Position-based variations
    for chinese_name, english_name in drinks_mapping.items():
        for pos in positions:
            categorized_prompts[chinese_name].extend([
                f"Pick {english_name} from {pos}",
                f"Get {english_name} off {pos}",
                f"Take {english_name} from the {pos}",
                f"Grab {english_name} on {pos}"
            ])
            
    # 4. Target variations
    for chinese_name, english_name in drinks_mapping.items():
        for target in targets:
            categorized_prompts[chinese_name].extend([
                f"Move {english_name} to {target}",
                f"Place {english_name} in {target}",
                f"Put {english_name} into {target}",
                f"Transfer {english_name} to {target}"
            ])

    # 5. Object type variations
    for chinese_name, english_name in drinks_mapping.items():
        color = color_map[english_name]
        for obj_type in object_types:
            categorized_prompts[chinese_name].extend([
                f"Pick {color} {obj_type}",
                f"Grab the {color} {obj_type}",
                f"Take {color} {obj_type} to box",
                f"Get {color} {obj_type} from shelf"
            ])
    
    # 6. Specific brand references
    brand_mapping = {
        "冰红茶": ["Pick Kangshifu red tea", "Get Kangshifu red tea", "Take Kangshifu red tea"],
        "青提绿茶": ["Pick Kangshifu green tea", "Get Kangshifu green tea", "Take Kangshifu green tea"], 
        "乳酸菌水乐": ["Pick Kangshifu lactic drink", "Get Kangshifu lactic drink", "Take Kangshifu lactic drink"],
        "劲凉冰红茶": ["Pick Kangshifu cool tea", "Get Kangshifu cool tea", "Take Kangshifu cool tea"]
    }
    
    for chinese_name, brand_prompts in brand_mapping.items():
        categorized_prompts[chinese_name].extend(brand_prompts)
    
    # 7. Contrastive descriptions
    drinks_list = list(drinks_mapping.values())
    colors_list = list(color_map.values())
    
    # Drink vs drink negation
    for chinese_name, english_name in drinks_mapping.items():
        for other_chinese, other_english in drinks_mapping.items():
            if chinese_name != other_chinese:
                categorized_prompts[chinese_name].extend([
                    f"Pick {english_name}, not {other_english}",
                    f"Get {english_name} from shelf, not the {other_english}",
                    f"Take {english_name} to box, not {other_english}",
                    f"Grab {english_name} instead of {other_english}"
                ])
    
    # Color vs color negation  
    for chinese_name, english_name in drinks_mapping.items():
        color = color_map[english_name]
        for other_chinese, other_english in drinks_mapping.items():
            if chinese_name != other_chinese:
                other_color = color_map[other_english]
                categorized_prompts[chinese_name].extend([
                    f"Pick {color} bottle, not {other_color}",
                    f"Get {color} drink, not the {other_color} one",
                    f"Take {color} beverage, not {other_color}",
                    f"Grab the {color} one, not {other_color}"
                ])
    
    # Position + negation combinations
    for chinese_name, english_name in drinks_mapping.items():
        for pos in positions[:3]:  # Use fewer positions to avoid too many combinations
            for other_chinese, other_english in drinks_mapping.items():
                if chinese_name != other_chinese:
                    categorized_prompts[chinese_name].extend([
                        f"Pick {english_name} from {pos}, not the {other_english}",
                        f"Get {english_name} off {pos}, not {other_english}"
                    ])
    
    # Remove duplicates within each category
    for chinese_name in categorized_prompts:
        categorized_prompts[chinese_name] = list(set(categorized_prompts[chinese_name]))
    
    return categorized_prompts

def save_prompts_to_json(categorized_prompts: Dict[str, List[str]], output_path: str = "drink_prompts.json"):
    """Save categorized prompts to a JSON file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(categorized_prompts, f, ensure_ascii=False, indent=2)
    
    print(f"Saved categorized prompts to {output_path}")
    return output_path

# Generate all prompts (legacy function for backward compatibility)
def generate_english_prompts():
    """Generate diverse English prompts for drink picking tasks (legacy function)"""
    categorized_prompts = generate_categorized_english_prompts()
    all_prompts = []
    for prompts_list in categorized_prompts.values():
        all_prompts.extend(prompts_list)
    return list(set(all_prompts))

# Generate categorized prompts and save to JSON
if __name__ == "__main__":
    categorized_prompts = generate_categorized_english_prompts()
    
    # Save to JSON file
    json_path = save_prompts_to_json(categorized_prompts, "drink_prompts.json")
    
    print("=== CATEGORIZED PROMPTS SUMMARY ===")
    total_prompts = 0
    for chinese_name, prompts_list in categorized_prompts.items():
        english_name = {"冰红茶": "red tea", "青提绿茶": "green tea", "乳酸菌水乐": "lactic drink", "劲凉冰红茶": "cool blue tea"}[chinese_name]
        print(f"{chinese_name} ({english_name}): {len(prompts_list)} prompts")
        total_prompts += len(prompts_list)
        
        # Show first few examples
        print("  Examples:")
        for i, prompt in enumerate(prompts_list[:3], 1):
            print(f"    {i}. {prompt}")
        if len(prompts_list) > 3:
            print(f"    ... and {len(prompts_list) - 3} more")
        print()
    
    print(f"=== TOTAL: {total_prompts} prompts across {len(categorized_prompts)} drink categories ===") 