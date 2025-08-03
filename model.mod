
set GRADES;
set COMPONENTS;
set PROPERTIES;

param price{GRADES};
param min_volume{GRADES};
param max_volume{GRADES};

param cost{COMPONENTS};
param max_availability{COMPONENTS};
param min_comp_requirement{COMPONENTS};

param prop_value{COMPONENTS, PROPERTIES};
param spec_min{PROPERTIES, GRADES};
param spec_max{PROPERTIES, GRADES};

var blend{g in GRADES, c in COMPONENTS} >= 0;

maximize Total_Profit:
    sum{g in GRADES} (
        price[g] * sum{c in COMPONENTS} blend[g, c] - 
        sum{c in COMPONENTS} cost[c] * blend[g, c]
    );

s.t. Min_Volume{g in GRADES}:
    sum{c in COMPONENTS} blend[g, c] >= min_volume[g];

s.t. Max_Volume{g in GRADES}:
    sum{c in COMPONENTS} blend[g, c] <= max_volume[g];

s.t. Component_Availability{c in COMPONENTS}:
    sum{g in GRADES} blend[g, c] <= max_availability[c];

s.t. Component_Min_Requirement{c in COMPONENTS}:
    sum{g in GRADES} blend[g, c] >= min_comp_requirement[c];

s.t. Property_Min{p in PROPERTIES, g in GRADES}:
    sum{c in COMPONENTS} prop_value[c, p] * blend[g, c] >= spec_min[p, g] * sum{c in COMPONENTS} blend[g, c];

s.t. Property_Max{p in PROPERTIES, g in GRADES}:
    sum{c in COMPONENTS} prop_value[c, p] * blend[g, c] <= spec_max[p, g] * sum{c in COMPONENTS} blend[g, c];

solve;

end;