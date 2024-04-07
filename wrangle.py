# Convert the bins to categories from 1 - 10 i.e 1st decile to 10th decile
def binned_Inc_to_num(data):
    data = data.copy()
    # Mapping
    category_to_number = {
        "[22640, 34218.1]": 1,
        "(34218.1, 37413.8]": 2,
        "(37413.8, 40362.7]": 3,
        "(40362.7, 42724.4]": 4,
        "(42724.4, 45201]": 5,
        "(45201, 48021.6]": 6,
        "(48021.6, 51046.4]": 7,
        "(51046.4, 54545.6]": 8,
        "(54545.6, 61494.5]": 9,
        "(61494.5, 125635]": 10
    }
     
    data["binnedInc"] = data["binnedInc"].map(category_to_number)
    
    return data



# split geography into county and state
def geography_split(data):
    # Create a copy of the original data
    data = data.copy()
    # Split and expand at ','
    data[["county", "state"]] = data["Geography"].str.split(",", expand=True)
    # Drop the expanded data
    data = data.drop(columns="Geography")
    
    return data