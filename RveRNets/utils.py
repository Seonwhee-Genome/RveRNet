def translate_list_to_str(List):
    return ', '.join(str(cls) for cls in List)


def remove_item_by_value(dictionary, value_to_remove):
    keys_to_remove = [key for key, value in dictionary.items() if value == value_to_remove]
    for key in keys_to_remove:
        del dictionary[key]