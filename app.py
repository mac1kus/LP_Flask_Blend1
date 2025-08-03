from flask import Flask, render_template, request, send_file
from pulp import *
import math
from datetime import datetime
import subprocess
import os
import io
import tempfile
import traceback
from jinja2 import Environment, FileSystemLoader, select_autoescape, StrictUndefined

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration - Dynamically set file paths based on environment ---
# Check if running in a Render environment (or similar containerized platform)
if os.environ.get("RENDER"):
    BASE_PATH = "/tmp"
    print("Running on Render. Using /tmp for file storage.")
else:
    BASE_PATH = "."
    print("Running locally. Using current directory for file storage.")

RESULT_FILE_NAME = os.path.join(BASE_PATH, "result1.txt")
RANGE_REPORT_FILE_NAME = os.path.join(BASE_PATH, "result2.txt")
INFEASIBILITY_FILE_NAME = os.path.join(BASE_PATH, "infeasibility_analysis.txt")
MOD_FILE = os.path.join(BASE_PATH, "model.mod")
DAT_FILE = os.path.join(BASE_PATH, "data.dat")
GLPSOL_PATH = None

# --- Helper functions for conversions ---
def calculate_roi(ron):
    """Calculate ROI from RON using the given formula"""
    if ron < 85:
        return ron + 11.5
    else:
        return math.exp((0.0135 * ron) + 3.42)

def calculate_moi(mon):
    """Calculate MOI from MON using the given formula"""
    if mon < 85:
        return mon + 11.5
    else:
        return math.exp((0.0135 * mon) + 3.42)

def calculate_rvi(rvp):
    """Calculate RVI from RVP using the given formula"""
    return (rvp * 14.5) ** 1.25

def reverse_roi_to_ron(roi):
    """Convert ROI back to RON"""
    if roi > 96.5: 
        return (math.log(roi) - 3.42) / 0.0135
    else:
        return roi - 11.5

def reverse_moi_to_mon(moi):
    """Convert MOI back to MON"""
    if moi > 96.5: 
        return (math.log(moi) - 3.42) / 0.0135
    else:
        return moi - 11.5

def reverse_rvi_to_rvp(rvi):
    """Convert RVI back to RVP"""
    return (rvi ** (1/1.25)) / 14.5

def convert_component_properties(components_data):
    """Convert component properties from RVP/MON/RON to RVI/MOI/ROI"""
    for comp in components_data:
        properties = comp['properties']
        if 'RON' in properties:
            properties['ROI'] = calculate_roi(properties['RON'])
        if 'MON' in properties:
            properties['MOI'] = calculate_moi(properties['MON'])
        if 'RVP' in properties:
            properties['RVI'] = calculate_rvi(properties['RVP'])
    return components_data

def convert_specs_to_internal(specs_data):
    """Convert specification bounds from RVP/MON/RON to RVI/MOI/ROI"""
    converted_specs = specs_data.copy()
    if 'RON' in specs_data:
        converted_specs['ROI'] = {}
        for grade, bounds in specs_data['RON'].items():
            min_roi = calculate_roi(bounds['min']) if bounds['min'] != 0 and not math.isinf(bounds['min']) else bounds['min']
            max_roi = calculate_roi(bounds['max']) if not math.isinf(bounds['max']) else bounds['max']
            converted_specs['ROI'][grade] = {'min': min_roi, 'max': max_roi}
    
    if 'MON' in specs_data:
        converted_specs['MOI'] = {}
        for grade, bounds in specs_data['MON'].items():
            min_moi = calculate_moi(bounds['min']) if bounds['min'] != 0 and not math.isinf(bounds['min']) else bounds['min']
            max_moi = calculate_moi(bounds['max']) if not math.isinf(bounds['max']) else bounds['max']
            converted_specs['MOI'][grade] = {'min': min_moi, 'max': max_moi}
    
    if 'RVP' in specs_data:
        converted_specs['RVI'] = {}
        for grade, bounds in specs_data['RVP'].items():
            min_rvi = calculate_rvi(bounds['min']) if bounds['min'] != 0 and not math.isinf(bounds['min']) else bounds['min']
            max_rvi = calculate_rvi(bounds['max']) if not math.isinf(bounds['max']) else bounds['max']
            converted_specs['RVI'][grade] = {'min': min_rvi, 'max': max_rvi}
    return converted_specs

def write_timestamp_header_to_stringio(file_handle, title):
    """Write a standardized timestamp header to a StringIO object."""
    now = datetime.now()
    file_handle.write("=" * 80 + "\n")
    file_handle.write(f"{title}\n")
    file_handle.write("=" * 80 + "\n")
    file_handle.write(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
    file_handle.write(f"Report Date: {now.strftime('%A, %B %d, %Y')}\n")
    file_handle.write(f"Generation Time: {now.strftime('%I:%M:%S %p')}\n")
    file_handle.write("=" * 80 + "\n\n")

def prepare_specs_for_template(specs_data):
    """Prepare specs data for Jinja2 template by converting inf values to large numbers"""
    prepared_specs = {}
    for prop, grades in specs_data.items():
        prepared_specs[prop] = {}
        for grade, bounds in grades.items():
            prepared_specs[prop][grade] = {
                'min': bounds['min'] if not math.isinf(bounds['min']) else 0,
                'max': bounds['max'] if not math.isinf(bounds['max']) else 999999
            }
    return prepared_specs

def make_glpk_safe_name(name):
    """Convert names to GLPK-safe identifiers by replacing spaces with underscores"""
    return name.replace(' ', '_').replace('-', '_')

def analyze_grade_infeasibility(grade_name, grade_idx, grades_data, components_data, properties_list, specs_data, original_specs_data, spec_bounds):
    """Enhanced infeasibility analysis that finds multiple feasible paths to fix constraints"""
    diagnostics = []
    diagnostics.append(f"ENHANCED INFEASIBILITY ANALYSIS FOR {grade_name}")
    diagnostics.append("=" * 70)
    
    grade_min = grades_data[grade_idx]['min']
    grade_max = grades_data[grade_idx]['max']
    grade_price = grades_data[grade_idx]['price']
    
    components = [c['name'] for c in components_data]
    component_cost = {c['name']: c['cost'] for c in components_data}
    component_availability = {c['name']: c['availability'] for c in components_data}
    component_min_comp = {c['name']: c['min_comp'] for c in components_data}
    
    property_value = {}
    for comp_data in components_data:
        for prop in properties_list:
            property_value[(prop, comp_data['name'])] = comp_data['properties'].get(prop, 0.0)

    def create_test_model(relaxed_constraints=None):
        """Create a test model with optionally relaxed constraints"""
        import time
        model_name = f"{grade_name}_Test_{int(time.time() * 1000000) % 1000000}"
        model = LpProblem(model_name, LpMaximize)
        
        # Create variables with unique names
        blend = {}
        for comp in components:
            var_name = f"Blend_{comp}_{int(time.time() * 1000000) % 1000000}"
            blend[comp] = LpVariable(var_name, lowBound=0, cat='Continuous')
        
        # Objective
        model += (
            grade_price * lpSum([blend[comp] for comp in components]) - 
            lpSum([component_cost[comp] * blend[comp] for comp in components])
        ), "Profit"
        
        # Volume constraints
        total = lpSum([blend[comp] for comp in components])
        model += total >= grade_min, f"{grade_name}_Min_{int(time.time() * 1000000) % 1000000}"
        model += total <= grade_max, f"{grade_name}_Max_{int(time.time() * 1000000) % 1000000}"
        
        # Component availability
        for comp in components:
            constraint_name = f"{comp}_Availability_{int(time.time() * 1000000) % 1000000}"
            model += blend[comp] <= component_availability[comp], constraint_name
        
        # Component minimums
        for comp in components:
            min_comp_val = component_min_comp.get(comp, 0)
            if min_comp_val is not None and min_comp_val > 0:
                constraint_name = f"{comp}_Min_{int(time.time() * 1000000) % 1000000}"
                model += blend[comp] >= min_comp_val, constraint_name
        
        # Property constraints (with optional relaxation)
        for prop in properties_list:
            min_val, max_val = spec_bounds.get((prop, grade_name), (0.0, float('inf')))
            
            # Apply relaxation if specified
            if relaxed_constraints:
                for relax_prop, relax_type, relax_amount in relaxed_constraints:
                    if prop == relax_prop:
                        if relax_type == 'min':
                            min_val = max(0, min_val - relax_amount)
                        elif relax_type == 'max':
                            max_val = max_val + relax_amount if not math.isinf(max_val) else float('inf')
            
            if min_val is not None and not math.isinf(min_val) and min_val > 0:
                weighted_sum = lpSum([property_value.get((prop, comp), 0) * blend[comp] for comp in components])
                constraint_name = f"{grade_name}_{prop}_Min_{int(time.time() * 1000000) % 1000000}"
                model += weighted_sum >= min_val * total, constraint_name
            
            if max_val is not None and not math.isinf(max_val):
                weighted_sum = lpSum([property_value.get((prop, comp), 0) * blend[comp] for comp in components])
                constraint_name = f"{grade_name}_{prop}_Max_{int(time.time() * 1000000) % 1000000}"
                model += weighted_sum <= max_val * total, constraint_name
        
        return model, blend, total
    
    def get_display_property_info(prop, value):
        """Convert internal property values to display values"""
        if prop == 'ROI':
            return 'RON', reverse_roi_to_ron(value)
        elif prop == 'MOI':
            return 'MON', reverse_moi_to_mon(value)
        elif prop == 'RVI':
            return 'RVP', reverse_rvi_to_rvp(value)
        else:
            return prop, value
    
    def calculate_achieved_properties(blend_vars, total_volume):
        """Calculate all achieved properties for a given blend"""
        achieved = {}
        for check_prop in ['SPG', 'SUL', 'RON', 'MON', 'RVP', 'E70', 'E10', 'E15', 'ARO', 'BEN', 'OXY', 'OLEFIN']:
            if check_prop in ['RON', 'MON', 'RVP']:
                # These need conversion from internal units
                if check_prop == 'RON':
                    internal_prop = 'ROI'
                elif check_prop == 'MON':
                    internal_prop = 'MOI'
                elif check_prop == 'RVP':
                    internal_prop = 'RVI'
                
                weighted_sum = sum(property_value.get((internal_prop, comp), 0) * (blend_vars[comp].varValue or 0) for comp in components)
                avg_internal = weighted_sum / total_volume if total_volume > 0 else 0
                
                if check_prop == 'RON':
                    achieved[check_prop] = reverse_roi_to_ron(avg_internal)
                elif check_prop == 'MON':
                    achieved[check_prop] = reverse_moi_to_mon(avg_internal)
                elif check_prop == 'RVP':
                    achieved[check_prop] = reverse_rvi_to_rvp(avg_internal)
            else:
                weighted_sum = sum(property_value.get((check_prop, comp), 0) * (blend_vars[comp].varValue or 0) for comp in components)
                achieved[check_prop] = weighted_sum / total_volume if total_volume > 0 else 0
        
        return achieved
    
    # First, verify it's actually infeasible
    try:
        base_model, base_blend, base_total = create_test_model()
        base_model.solve(PULP_CBC_CMD(msg=0))
        
        if base_model.status == LpStatusOptimal:
            diagnostics.append("ERROR: Model is actually feasible! No analysis needed.")
            return diagnostics
        
        diagnostics.append("1. CONFIRMED: Model is infeasible as stated")
        diagnostics.append("")
        
        # Get all active constraints
        active_constraints = []
        constraint_details = {}
        
        for prop in properties_list:
            min_val, max_val = spec_bounds.get((prop, grade_name), (0.0, float('inf')))
            
            if min_val is not None and not math.isinf(min_val) and min_val > 0:
                constraint_key = (prop, 'min')
                active_constraints.append(constraint_key)
                
                # Convert back for display
                display_prop, display_val = get_display_property_info(prop, min_val)
                constraint_details[constraint_key] = f"{display_prop} >= {display_val:.3f}"
            
            if max_val is not None and not math.isinf(max_val):
                constraint_key = (prop, 'max')
                active_constraints.append(constraint_key)
                
                # Convert back for display
                display_prop, display_val = get_display_property_info(prop, max_val)
                constraint_details[constraint_key] = f"{display_prop} <= {display_val:.3f}"
        
        # Test which single constraint removals make it feasible
        diagnostics.append("2. CRITICAL CONSTRAINT IDENTIFICATION")
        diagnostics.append("   (Testing which individual constraints cause infeasibility)")
        diagnostics.append("")
        
        critical_constraints = []
        
        for constraint_key in active_constraints:
            prop, bound_type = constraint_key
            
            # Create model without this specific constraint
            test_model, test_blend, test_total = create_test_model()
            
            # Clear and rebuild constraints without the tested one
            test_model.constraints = {}
            
            # Re-add volume constraints
            test_model += test_total >= grade_min, f"{grade_name}_Min_Test"
            test_model += test_total <= grade_max, f"{grade_name}_Max_Test"
            
            # Re-add component availability constraints
            for comp in components:
                test_model += test_blend[comp] <= component_availability[comp], f"{comp}_Availability_Test"
            
            # Re-add component minimum constraints
            for comp in components:
                min_comp_val = component_min_comp.get(comp, 0)
                if min_comp_val is not None and min_comp_val > 0:
                    test_model += test_blend[comp] >= min_comp_val, f"{comp}_Min_Test"
            
            # Re-add property constraints except the one we're testing
            for test_prop in properties_list:
                test_min_val, test_max_val = spec_bounds.get((test_prop, grade_name), (0.0, float('inf')))
                
                # Skip the constraint we're testing
                if test_prop == prop and bound_type == 'min':
                    continue
                if test_prop == prop and bound_type == 'max':
                    continue
                
                if test_min_val is not None and not math.isinf(test_min_val) and test_min_val > 0:
                    weighted_sum = lpSum([property_value.get((test_prop, comp), 0) * test_blend[comp] for comp in components])
                    test_model += weighted_sum >= test_min_val * test_total, f"{grade_name}_{test_prop}_Min_Test"
                
                if test_max_val is not None and not math.isinf(test_max_val):
                    weighted_sum = lpSum([property_value.get((test_prop, comp), 0) * test_blend[comp] for comp in components])
                    test_model += weighted_sum <= test_max_val * test_total, f"{grade_name}_{test_prop}_Max_Test"
            
            test_model.solve(PULP_CBC_CMD(msg=0))
            
            if test_model.status == LpStatusOptimal:
                critical_constraints.append(constraint_key)
                constraint_desc = constraint_details[constraint_key]
                diagnostics.append(f"   ✗ CRITICAL: {constraint_desc}")
                
                # Get the achieved value for this property
                total_vol = sum(test_blend[comp].varValue or 0 for comp in components)
                if total_vol > 0:
                    display_prop, achieved_val = get_display_property_info(prop, 
                        sum(property_value.get((prop, comp), 0) * (test_blend[comp].varValue or 0) for comp in components) / total_vol)
                    diagnostics.append(f"       Without this constraint, {display_prop} = {achieved_val:.3f}")
        
        if not critical_constraints:
            diagnostics.append("   No single constraint removal makes it feasible.")
            diagnostics.append("   This indicates complex multi-constraint interactions.")
        
        diagnostics.append("")
        diagnostics.append("FEASIBILITY SUMMARY & RECOMMENDATIONS")
        diagnostics.append("=" * 50)
        
        if critical_constraints:
            diagnostics.append("CRITICAL CONSTRAINTS (removing any one makes problem feasible):")
            for constraint_key in critical_constraints:
                desc = constraint_details[constraint_key]
                diagnostics.append(f"• {desc}")
        else:
            diagnostics.append("Multiple constraints interact to cause infeasibility.")
            diagnostics.append("Consider relaxing several constraints simultaneously.")
    
    except Exception as e:
        diagnostics.append(f"Error during infeasibility analysis: {str(e)}")
        diagnostics.append("This may indicate a deeper issue with the model setup.")
    
    return diagnostics

# --- Core LP Optimization Logic ---
def run_optimization(grades_data, components_data, properties_list, specs_data, solver_choice):
    # Store original specs_data for diagnostics
    original_specs_data = specs_data.copy()
    
    components_data = convert_component_properties(components_data)
    specs_data = convert_specs_to_internal(specs_data)
    
    grades = [g['name'] for g in grades_data]
    barrel_min = [g['min'] for g in grades_data]
    barrel_max = [g['max'] for g in grades_data]
    gasoline_price = [g['price'] for g in grades_data]

    components = [c['name'] for c in components_data]
    
    if not components:
        raise ValueError("No components found for optimization. Please check input data from the form.")

    component_cost = {c['name']: c['cost'] for c in components_data}
    component_availability = {c['name']: c['availability'] for c in components_data}
    component_min_comp = {c['name']: c['min_comp'] for c in components_data}

    property_value = {}
    for comp_data in components_data:
        for prop in properties_list:
            property_value[(prop, comp_data['name'])] = comp_data['properties'].get(prop, 0.0)

    spec_bounds = {}
    for prop_name, grade_specs in specs_data.items():
        for grade_name, bounds in grade_specs.items():
            spec_bounds[(prop_name, grade_name)] = (bounds['min'], bounds['max'])

    model = LpProblem("Gasoline_Blending", LpMaximize)
    blend = LpVariable.dicts("Blend", (grades, components), lowBound=0, cat='Continuous')

    model += lpSum([
        gasoline_price[i] * lpSum([blend[grades[i]][comp] for comp in components]) - 
        lpSum([component_cost[comp] * blend[grades[i]][comp] for comp in components])
        for i in range(len(grades)) 
    ]), "Total_Profit"

    for i in range(len(grades)): 
        total = lpSum([blend[grades[i]][comp] for comp in components])
        model += total >= barrel_min[i], f"{grades[i]}_Min"
        model += total <= barrel_max[i], f"{grades[i]}_Max"

    for g in grades:
        total_blend = lpSum([blend[g][comp] for comp in components])
        for p in properties_list:
            weighted_sum = lpSum([
                property_value.get((p, comp), 0) * blend[g][comp] for comp in components
            ])
            min_val, max_val = spec_bounds.get((p, g), (0.0, float('inf')))

            if min_val is not None and not math.isinf(min_val) and not math.isnan(min_val):
                model += weighted_sum >= min_val * total_blend, f"{g}_{p}_Min"
            if max_val is not None and not math.isinf(max_val) and not math.isnan(max_val):
                model += weighted_sum <= max_val * total_blend, f"{g}_{p}_Max"
    
    for comp in components:
        model += lpSum([blend[g][comp] for g in grades]) <= component_availability[comp], f"{comp}_Availability_Max"

    for comp in components:
        min_comp_val = component_min_comp.get(comp, 0)
        if min_comp_val is not None and min_comp_val > 0:
            model += lpSum([blend[g][comp] for g in grades]) >= min_comp_val, f"{comp}_Min_Comp"
    
    for g in grades:
        for comp in components:
            model += blend[g][comp] >= 0, f"{g}_{comp}_NonNegative"

    solver_used = ""
    if solver_choice == "GLPK":
        solver_used = "GLPK"
        try:
            solver = GLPK_CMD(msg=0, path=GLPSOL_PATH)
            model.solve(solver)
        except FileNotFoundError as e:
            solver_used = "CBC (Fallback)"
            model.solve(PULP_CBC_CMD(msg=0))
    else: 
        solver_used = "CBC"
        model.solve(PULP_CBC_CMD(msg=0))

    result1_content = io.StringIO()
    write_timestamp_header_to_stringio(result1_content, "GASOLINE BLENDING OPTIMIZATION REPORT")

    # Store overall status
    overall_status = LpStatus[model.status]
    result1_content.write("Overall Status: " + overall_status + "\n")
    result1_content.write(f"Solver Used: {solver_used}\n")
    if model.status == LpStatusOptimal:
        result1_content.write("Objective Value (Profit): {:.2f}\n".format(value(model.objective)))
    result1_content.write("\n")

    result1_content.write("=== Gasoline Grade Overview ===\n")
    grade_overview_data = [["GASOLINE", "MIN", "MAX", "PRICE"]]
    for i, grade in enumerate(grades):
        grade_overview_data.append([
            grade,
            f"{barrel_min[i]:.0f}",
            f"{barrel_max[i]:.0f}",
            f"{gasoline_price[i]:.0f}"
        ])

    overview_column_widths = [max(len(str(item)) for item in col) for col in zip(*grade_overview_data)]
    result1_content.write("| " + " | ".join(grade_overview_data[0][i].ljust(overview_column_widths[i]) for i in range(len(grade_overview_data[0]))) + " |\n")
    separator_parts = [("-" * width) for width in overview_column_widths]
    result1_content.write("|-" + "-|-".join(separator_parts) + "-|\n")
    for row_content in grade_overview_data[1:]:
        formatted_row = [str(row_content[0]).ljust(overview_column_widths[0]), str(row_content[1]).rjust(overview_column_widths[1]), str(row_content[2]).rjust(overview_column_widths[2]), str(row_content[3]).rjust(overview_column_widths[3])]
        result1_content.write("| " + " | ".join(formatted_row) + " |\n")
    result1_content.write("\n")

    def format_spec_value_concise(val):
        if val is None: return "N/A"
        if math.isinf(val): return "inf"
        if math.isnan(val): return "NaN"
        return f"{val:g}"

    display_properties_list = ["SPG", "SUL", "RON","ROI","MON","MOI","RVP","RVI","E70", "E10", "E15", "ARO", "BEN", "OXY", "OLEFIN"]
    
    # Dictionary to store grade-specific results
    grade_results = {}
    infeasibility_report_stringio = io.StringIO()
    write_timestamp_header_to_stringio(infeasibility_report_stringio, "GRADE INFEASIBILITY ANALYSIS REPORT")
    has_infeasible_grades = False
    
    # If overall solution is infeasible, try to solve for each grade individually
    if model.status != LpStatusOptimal:
        for current_grade_idx, current_grade in enumerate(grades):
            # Create a model for just this grade
            single_model = LpProblem(f"{current_grade}_Only", LpMaximize)
            single_blend = LpVariable.dicts("Blend", components, lowBound=0, cat='Continuous')
            
            # Objective: maximize profit for this grade only
            single_model += (
                gasoline_price[current_grade_idx] * lpSum([single_blend[comp] for comp in components]) - 
                lpSum([component_cost[comp] * single_blend[comp] for comp in components])
            ), "Profit"
            
            # Volume constraints
            total = lpSum([single_blend[comp] for comp in components])
            single_model += total >= barrel_min[current_grade_idx], f"{current_grade}_Min"
            single_model += total <= barrel_max[current_grade_idx], f"{current_grade}_Max"
            
            # Property constraints
            for p in properties_list:
                weighted_sum = lpSum([
                    property_value.get((p, comp), 0) * single_blend[comp] for comp in components
                ])
                min_val, max_val = spec_bounds.get((p, current_grade), (0.0, float('inf')))
                
                if min_val is not None and not math.isinf(min_val) and not math.isnan(min_val):
                    single_model += weighted_sum >= min_val * total, f"{current_grade}_{p}_Min"
                if max_val is not None and not math.isinf(max_val) and not math.isnan(max_val):
                    single_model += weighted_sum <= max_val * total, f"{current_grade}_{p}_Max"
            
            # Component availability
            for comp in components:
                single_model += single_blend[comp] <= component_availability[comp], f"{comp}_Availability"
            
            # Component minimums
            for comp in components:
                min_comp_val = component_min_comp.get(comp, 0)
                if min_comp_val is not None and min_comp_val > 0:
                    single_model += single_blend[comp] >= min_comp_val, f"{comp}_Min"
            
            # Solve
            single_model.solve(PULP_CBC_CMD(msg=0))
            
            grade_results[current_grade] = {
                'status': LpStatus[single_model.status],
                'model': single_model,
                'blend': single_blend,
                'profit': value(single_model.objective) if single_model.status == LpStatusOptimal else 0
            }
    else:
        # If overall is optimal, all grades are optimal
        for current_grade in grades:
            grade_results[current_grade] = {
                'status': 'Optimal',
                'model': model,
                'blend': blend[current_grade],
                'profit': 0
            }
    
    # Now display results for each grade
    for current_grade_idx, current_grade in enumerate(grades):
        grade_selling_price = gasoline_price[current_grade_idx]
        result1_content.write(f"\n{'='*60}\n")
        result1_content.write(f"{current_grade} GASOLINE\n")
        result1_content.write(f"{'='*60}\n")
        result1_content.write(f"Status: {grade_results[current_grade]['status']}\n")
        result1_content.write(f"Price: ${grade_selling_price:.2f}/bbl\n")
        
        # If this grade is infeasible, show why
        if grade_results[current_grade]['status'] != 'Optimal':
            result1_content.write("\nINFEASIBILITY ANALYSIS:\n")
            diagnostics = analyze_grade_infeasibility(
                current_grade, 
                current_grade_idx,
                grades_data,
                components_data, 
                properties_list, 
                specs_data, 
                original_specs_data,
                spec_bounds
            )
            
            result1_content.write("See infeasibility_analysis.txt for detailed analysis\n\n")
            
            has_infeasible_grades = True
            for diag in diagnostics:
                infeasibility_report_stringio.write(diag + "\n")
            infeasibility_report_stringio.write("\n" + "="*80 + "\n\n")
            continue
        
        result1_content.write(f"\n=== Calculated Properties of '{current_grade}' Optimized Blend ===\n")
        
        if model.status == LpStatusOptimal:
            current_blend = blend[current_grade]
        else:
            current_blend = grade_results[current_grade]['blend']
        
        current_total_volume = sum(current_blend[comp].varValue or 0 for comp in components)
        current_grade_total_cost = sum(component_cost[comp] * (current_blend[comp].varValue or 0) for comp in components)
        current_grade_revenue = grade_selling_price * current_total_volume
        current_grade_profit = current_grade_revenue - current_grade_total_cost
        
        result1_content.write(f"Total Volume: {current_total_volume:.2f} bbl\n")
        result1_content.write(f"Total Cost: ${current_grade_total_cost:.2f}\n")
        result1_content.write(f"Total Revenue: ${current_grade_revenue:.2f}\n")
        result1_content.write(f"Profit: ${current_grade_profit:.2f}\n\n")

        table_data_for_printing = [["Component Name", "Vol(bbl)", "Cost($)"] + display_properties_list]
        for comp in components: 
            vol = current_blend[comp].varValue or 0
            comp_cost_val = component_cost[comp]
            row = [comp, f"{vol:.2f}", f"{comp_cost_val:.2f}"]
            for p in display_properties_list:
                if p == 'RON':
                    roi_val = property_value.get(('ROI', comp), 0)
                    val = reverse_roi_to_ron(roi_val) if roi_val > 0 else 0
                elif p == 'MON':
                    moi_val = property_value.get(('MOI', comp), 0)
                    val = reverse_moi_to_mon(moi_val) if moi_val > 0 else 0
                elif p == 'RVP':
                    rvi_val = property_value.get(('RVI', comp), 0)
                    val = reverse_rvi_to_rvp(rvi_val) if rvi_val > 0 else 0
                else:
                    val = property_value.get((p, comp), 0)
                
                row.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
            table_data_for_printing.append(row)

        combined_total_row_content = ["TOTAL", f"{current_total_volume:.2f}", f"{current_grade_total_cost:.2f}"] + [""] * len(display_properties_list)
        quality_row_content = ["QUALITY", "", ""] 
        for p in display_properties_list:
            if p == 'RON':
                weighted_sum_roi = sum(property_value.get(('ROI', comp), 0) * (current_blend[comp].varValue or 0) for comp in components)
                avg_roi = weighted_sum_roi / current_total_volume if current_total_volume > 0 else 0
                calculated_property_value = reverse_roi_to_ron(avg_roi) if avg_roi > 0 else 0
            elif p == 'MON':
                weighted_sum_moi = sum(property_value.get(('MOI', comp), 0) * (current_blend[comp].varValue or 0) for comp in components)
                avg_moi = weighted_sum_moi / current_total_volume if current_total_volume > 0 else 0
                calculated_property_value = reverse_moi_to_mon(avg_moi) if avg_moi > 0 else 0
            elif p == 'RVP':
                weighted_sum_rvi = sum(property_value.get(('RVI', comp), 0) * (current_blend[comp].varValue or 0) for comp in components)
                avg_rvi = weighted_sum_rvi / current_total_volume if current_total_volume > 0 else 0
                calculated_property_value = reverse_rvi_to_rvp(avg_rvi) if avg_rvi > 0 else 0
            else:
                weighted_sum_for_grade = sum(property_value.get((p, comp), 0) * (current_blend[comp].varValue or 0) for comp in components)
                calculated_property_value = weighted_sum_for_grade / current_total_volume if current_total_volume > 0 else 0
            quality_row_content.append(f"{calculated_property_value:.4f}")
        
        spec_row_content = ["SPEC", "", ""] 
        for p in display_properties_list:
            if p in ['ROI', 'MOI', 'RVI']:
                min_spec_val, max_spec_val = spec_bounds.get((p, current_grade), (0, float('inf')))
            else:
                min_spec_val, max_spec_val = original_specs_data.get(p, {}).get(current_grade, {"min": 0, "max": float('inf')}).values()
            formatted_lb_spec = format_spec_value_concise(min_spec_val)
            formatted_ub_spec = format_spec_value_concise(max_spec_val)
            spec_string = f"{formatted_lb_spec}-{formatted_ub_spec}"
            spec_row_content.append(spec_string)

        all_rows_for_width_calc = [table_data_for_printing[0]] + table_data_for_printing[1:] + [combined_total_row_content, quality_row_content, spec_row_content]
        column_widths = [max(len(str(item)) for item in col) for col in zip(*all_rows_for_width_calc)]

        def write_formatted_row(data, alignment):
            formatted_row = [str(data[0]).ljust(column_widths[0])] + [str(data[1]).ljust(column_widths[1])] + [str(data[2]).ljust(column_widths[2])]
            for i, item in enumerate(data[3:]):
                formatted_row.append(str(item).rjust(column_widths[i+3]))
            result1_content.write("| " + " | ".join(formatted_row) + " |\n")

        header_row = table_data_for_printing[0]
        result1_content.write("| " + " | ".join(header_row[i].ljust(column_widths[i]) for i in range(len(header_row))) + " |\n")
        separator_parts = [("-" * width) for width in column_widths]
        result1_content.write("|-" + "-|-".join(separator_parts) + "-|\n")
        for row_content in table_data_for_printing[1:]:
            write_formatted_row(row_content, 'right')
        write_formatted_row(combined_total_row_content, 'right')
        write_formatted_row(quality_row_content, 'right')
        write_formatted_row(spec_row_content, 'right')
    
    result1_content.write("\n\n=== Component Summary ===\n")
    component_summary_data = [["Component", "Available (bbl)", "Used (bbl)"]]
    for comp in components:
        if model.status == LpStatusOptimal:
            total_used_volume = sum(blend[g][comp].varValue or 0 for g in grades)
        else:
            total_used_volume = 0
            for g in grades:
                if grade_results[g]['status'] == 'Optimal' and grade_results[g]['blend'] is not None:
                    total_used_volume += grade_results[g]['blend'][comp].varValue or 0
        
        available_quantity = component_availability.get(comp, 0)
        component_summary_data.append([comp, f"{available_quantity:.2f}", f"{total_used_volume:.2f}"])

    summary_column_widths = [max(len(str(item)) for item in col) for col in zip(*component_summary_data)]
    result1_content.write("| " + " | ".join(component_summary_data[0][i].ljust(summary_column_widths[i]) for i in range(len(component_summary_data[0]))) + " |\n")
    separator_parts = [("-" * width) for width in summary_column_widths]
    result1_content.write("|-" + "-|-".join(separator_parts) + "-|\n")
    for row_content in component_summary_data[1:]:
        formatted_row = [str(row_content[0]).ljust(summary_column_widths[0]), str(row_content[1]).rjust(summary_column_widths[1]), str(row_content[2]).rjust(summary_column_widths[2])]
        result1_content.write("| " + " | ".join(formatted_row) + " |\n")
    
    result1_content.seek(0)
    
    range_report_content = io.StringIO()
    if solver_choice == "GLPK" and model.status == LpStatusOptimal:
        try:
            mod_file_path = MOD_FILE
            dat_file_path = DAT_FILE
            
            # --- MathProg File Generation ---
            env = Environment(
                loader=FileSystemLoader(BASE_PATH),
                undefined=StrictUndefined
            )
            
            mod_template_str = """
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
"""

            dat_template_str = """
set GRADES := {% for g in grades %}{{ make_glpk_safe_name(g.name) }} {% endfor %};
set COMPONENTS := {% for c in components %}{{ make_glpk_safe_name(c.name) }} {% endfor %};
set PROPERTIES := {% for p in properties %}{{ p }} {% endfor %};

param price := {% for g in grades %}{{ make_glpk_safe_name(g.name) }} {{ g.price }} {% endfor %};
param min_volume := {% for g in grades %}{{ make_glpk_safe_name(g.name) }} {{ g.min }} {% endfor %};
param max_volume := {% for g in grades %}{{ make_glpk_safe_name(g.name) }} {{ g.max }} {% endfor %};

param cost := {% for c in components %}{{ make_glpk_safe_name(c.name) }} {{ c.cost }} {% endfor %};
param max_availability := {% for c in components %}{{ make_glpk_safe_name(c.name) }} {{ c.availability }} {% endfor %};
param min_comp_requirement := {% for c in components %}{{ make_glpk_safe_name(c.name) }} {{ c.min_comp }} {% endfor %};

param prop_value: {% for p in properties %}{{ p }} {% endfor %} :=
{% for c in components %} {{ make_glpk_safe_name(c.name) }} {% for p in properties %}{{ c.properties.get(p, 0) }} {% endfor %}{% endfor %};

param spec_min: {% for g in grades %}{{ make_glpk_safe_name(g.name) }} {% endfor %} :=
{% for p in properties %} {{ p }} {% for g in grades %}{{ prepared_specs.get(p, {}).get(g.name, {}).get('min', 0) }} {% endfor %}{% endfor %};

param spec_max: {% for g in grades %}{{ make_glpk_safe_name(g.name) }} {% endfor %} :=
{% for p in properties %} {{ p }} {% for g in grades %}{{ prepared_specs.get(p, {}).get(g.name, {}).get('max', 999999) }} {% endfor %}{% endfor %};

end;
"""
            
            prepared_specs = prepare_specs_for_template(specs_data)
            from jinja2 import Template
            mod_template = Template(mod_template_str)
            dat_template = Template(dat_template_str)
            
            grades_raw = grades_data
            components_raw = components_data
            properties_raw = properties_list
            
            mod_output = mod_template.render()
            dat_output = dat_template.render(
                grades=grades_raw, 
                components=components_raw, 
                properties=properties_raw, 
                prepared_specs=prepared_specs,
                make_glpk_safe_name=make_glpk_safe_name
            )

            with open(mod_file_path, "w") as f:
                f.write(mod_output)
            with open(dat_file_path, "w") as f:
                f.write(dat_output)
            
            # --- Run GLPK Range Analysis ---
            range_output_file = os.path.join(BASE_PATH, "temp_range_output.txt")
            glpsol_range_command = ["glpsol", "--math", mod_file_path, "--data", dat_file_path, "--ranges", range_output_file]
            
            result = subprocess.run(glpsol_range_command, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(range_output_file):
                write_timestamp_header_to_stringio(range_report_content, "GLPK RANGE ANALYSIS REPORT")
                with open(range_output_file, 'r', encoding='utf-8') as temp_f:
                    range_report_content.write(temp_f.read())
            else:
                range_report_content = io.StringIO()
                write_timestamp_header_to_stringio(range_report_content, "GLPK RANGE ANALYSIS REPORT")
                range_report_content.write("GLPK Range Analysis is only available for GLPK solver with an Optimal solution.\n")
                
        except Exception as e:
            range_report_content = io.StringIO()
            write_timestamp_header_to_stringio(range_report_content, "GLPK RANGE ANALYSIS REPORT")
            range_report_content.write(f"Error during GLPK Range Analysis: {str(e)}\n")
            range_report_content.write("Range analysis is only available for GLPK solver with an Optimal solution.\n")
    else:
        range_report_content = io.StringIO()
        write_timestamp_header_to_stringio(range_report_content, "GLPK RANGE ANALYSIS REPORT")
        range_report_content.write("GLPK Range Analysis is only available for GLPK solver with an Optimal solution.\n")
    
    range_report_content.seek(0)
    
    # Finalize infeasibility report
    if not has_infeasible_grades:
        infeasibility_report_stringio.write("All grades were successfully optimized. No infeasibility issues found.\n")
    infeasibility_report_stringio.seek(0)

    return result1_content.getvalue(), range_report_content.getvalue(), infeasibility_report_stringio.getvalue()

# Main route handlers
@app.route('/', methods=['GET'])
def index():
    grades_initial = [{"name": "Regular", "min": 4000.000000, "max": 400000.000000, "price": 100.000000},
                      {"name": "Premium", "min": 0.000000, "max": 400000.000000, "price": 110.000000},
                      {"name": "Super Premium", "min": 0.000000, "max": 4000.000000, "price": 200.000000}]
    all_properties = ["SPG", "SUL", "RON","MON","RVP","E70", "E10", "E15", "ARO", "BEN", "OXY", "OLEFIN"]
    components_initial = [
        {"name": "C4B", "tag": "Alkyl Butane", "min_comp": 0.0, "availability": 1000000.000000, "factor": 1.300000, "cost": 130.000000, 
         "properties": {"SPG": 0.584400, "SUL": 0.000100, "RON": 93.800000, "MON": 89.600000, "RVP": 3.191000, "E70": 100.000000, "E10": 100.000000, "E15": 100.000000, "ARO": 0.000000, "BEN": 0.000000, "OXY": 0.000000, "OLEFIN": 0.000000}},
        {"name": "IS1", "tag": "Isomerate", "min_comp": 0.00, "availability": 1000000.000000, "factor": 1.250000, "cost": 125.000000, 
         "properties": {"SPG": 0.661000, "SUL": 0.500000, "RON": 88.560000, "MON": 86.150000, "RVP": 0.839000, "E70": 92.000000, "E10": 100.000000, "E15": 100.000000, "ARO": 0.000000, "BEN": 0.000000, "OXY": 0.000000, "OLEFIN": 0.000000}},
        {"name": "RFL", "tag": "Reformate", "min_comp": 0.00, "availability": 1000000.000000, "factor": 1.050000, "cost": 105.000000, 
         "properties": {"SPG": 0.819000, "SUL": 0.000000, "RON": 97.000000, "MON": 86.150000, "RVP": 0.139000, "E70": 0.001000, "E10": 4.000000, "E15": 72.250000, "ARO": 61.800000, "BEN": 0.438400, "OXY": 0.000000, "OLEFIN": 0.775600}},
        {"name": "F5X", "tag": "Mixed RFC", "min_comp": 0.00, "availability": 1000000.000000, "factor": 0.700000, "cost": 70.000000, 
         "properties": {"SPG": 0.644700, "SUL": 10.000000, "RON": 94.600000, "MON": 89.650000, "RVP": 1.310000, "E70": 100.000000, "E10": 100.000000, "E15": 100.000000, "ARO": 0.000000, "BEN": 1.160000, "OXY": 0.000000, "OLEFIN": 57.700000}},
        {"name": "RCG", "tag": "FCC Gasoline", "min_comp": 0, "availability": 1000000.000000, "factor": 0.900000, "cost": 90.000000, 
         "properties": {"SPG": 0.785600, "SUL": 20.000000, "RON": 94.430000, "MON": 82.440000, "RVP": 0.210000, "E70": 8.854800, "E10": 36.400000, "E15": 67.300000, "ARO": 50.400000, "BEN": 1.718300, "OXY": 0.000000, "OLEFIN": 19.670000}},
        {"name": "IC4", "tag": "DIB IC4", "min_comp": 0, "availability": 1000000.000000, "factor": 0.900000, "cost": 90.000000, 
         "properties": {"SPG": 0.563300, "SUL": 10.000000, "RON": 100.050000, "MON": 97.540000, "RVP": 4.347000, "E70": 100.000000, "E10": 100.000000, "E15": 100.000000, "ARO": 0.000000, "BEN": 0.000000, "OXY": 0.000000, "OLEFIN": 0.000000}},
        {"name": "HBY", "tag": "SHIP C4", "min_comp": 0, "availability": 1000000.000000, "factor": 0.750000, "cost": 75.000000, 
         "properties": {"SPG": 0.593600, "SUL": 10.000000, "RON": 98.200000, "MON": 89.000000, "RVP": 3.674000, "E70": 100.000000, "E10": 100.000000, "E15": 100.000000, "ARO": 0.000000, "BEN": 0.000000, "OXY": 0.000000, "OLEFIN": 60.800000}},
        {"name": "AKK", "tag": "Alkylate", "min_comp": 0.0, "availability": 1000000.000000, "factor": 0.700000, "cost": 70.000000, 
         "properties": {"SPG": 0.703200, "SUL": 0.000100, "RON": 76.130000, "MON": 92.000000, "RVP": 0.403000, "E70": 10.000000, "E10": 35.000000, "E15": 100.000000, "ARO": 0.000000, "BEN": 0.000000, "OXY": 0.000000, "OLEFIN": 0.000000}},
        {"name": "ETH", "tag": "Ethanol", "min_comp": 0.0, "availability": 1000000.000000, "factor": 0.750000, "cost": 75.000000, 
         "properties": {"SPG": 0.791000, "SUL": 1.000000, "RON": 128.000000, "MON": 100.000000, "RVP": 1.329000, "E70": 50.000000, "E10": 100.000000, "E15": 100.000000, "ARO": 0.000000, "BEN": 0.000000, "OXY": 34.780000, "OLEFIN": 0.000000}},
    ]
    regular_gasoline_price = next((g['price'] for g in grades_initial if g['name'] == 'Regular'), 100.00)
    for comp in components_initial:
        comp['display_cost'] = comp['factor'] * regular_gasoline_price
        comp['cost'] = comp['display_cost']
    specs_initial = {
        "SPG": {"Regular": {"min": 0.720000, "max": 0.780000}, "Premium": {"min": 0.720000, "max": 0.780000}, "Super Premium": {"min": 0.720000, "max": 0.780000}},
        "SUL": {"Regular": {"min": 0.000000, "max": 10.000000}, "Premium": {"min": 0.000000, "max": 10.000000}, "Super Premium": {"min": 0.000000, "max": 10.000000}},
        "RON": {"Regular": {"min": 91.000000, "max": float('inf')}, "Premium": {"min": 95.000000, "max": float('inf')}, "Super Premium": {"min": 98.000000, "max": float('inf')}},
        "MON": {"Regular": {"min": 82.000000, "max": float('inf')}, "Premium": {"min": 86.000000, "max": float('inf')}, "Super Premium": {"min": 89.000000, "max": float('inf')}},
        "RVP": {"Regular": {"min": 0.000000, "max": 0.700000}, "Premium": {"min": 0.000000, "max": 0.700000}, "Super Premium": {"min": 0.000000, "max": 0.700000}},
        "E70": {"Regular": {"min": 22.000000, "max": 48.000000}, "Premium": {"min": 22.000000, "max": 48.000000}, "Super Premium": {"min": 22.000000, "max": 48.000000}},
        "E10": {"Regular": {"min": 44.000000, "max": 70.000000}, "Premium": {"min": 44.000000, "max": 70.000000}, "Super Premium": {"min": 44.000000, "max": 70.000000}},
        "E15": {"Regular": {"min": 76.000000, "max": float('inf')}, "Premium": {"min": 76.000000, "max": float('inf')}, "Super Premium": {"min": 76.000000, "max": float('inf')}},
        "ARO": {"Regular": {"min": 0.000000, "max": 35.000000}, "Premium": {"min": 0.000000, "max": 35.000000}, "Super Premium": {"min": 0.000000, "max": 35.000000}},
        "BEN": {"Regular": {"min": 0.000000, "max": 1.000000}, "Premium": {"min": 0.000000, "max": 1.000000}, "Super Premium": {"min": 0.000000, "max": 1.000000}},
        "OXY": {"Regular": {"min": 0.000000, "max": 2.700000}, "Premium": {"min": 0.000000, "max": 2.700000}, "Super Premium": {"min": 0.000000, "max": 2.700000}},
        "OLEFIN": {"Regular": {"min": 0.000000, "max": 15.000000}, "Premium": {"min": 0.000000, "max": 15.000000}, "Super Premium": {"min": 0.000000, "max": 15.000000}},
    }
    current_datetime = datetime.now()
    return render_template('input.html', 
                            grades=grades_initial, 
                            components=components_initial, 
                            properties=all_properties, 
                            specs=specs_initial,
                            current_datetime=current_datetime)

@app.route('/run_lp', methods=['POST'])
def run_lp():
    try:
        print("=== Starting LP Optimization ===")
        
        grades_data = []
        for i, grade_name in enumerate(["Regular", "Premium", "Super Premium"]):
            try:
                min_val_str = request.form.get(f'grade_{grade_name}_min', '0').strip()
                min_val = float(min_val_str) if min_val_str else 0.0
                
                max_val_str = request.form.get(f'grade_{grade_name}_max', '0').strip()
                max_val = float(max_val_str) if max_val_str else 0.0
                
                price_val_str = request.form.get(f'grade_{grade_name}_price', '0').strip()
                price_val = float(price_val_str) if price_val_str else 0.0
                
                grades_data.append({"name": grade_name, "min": min_val, "max": max_val, "price": price_val})
                print(f"Grade {grade_name}: min={min_val}, max={max_val}, price={price_val}")
            except ValueError as e:
                error_msg = f"Invalid input for {grade_name} grade: {e}"
                print(f"ERROR: {error_msg}")
                return error_msg, 400

        regular_gasoline_price = next((g['price'] for g in grades_data if g['name'] == 'Regular'), 100.00)
        components_data = []
        component_html_keys = ["C4B","IS1","RFL","F5X","RCG","IC4","HBY","AKK","ETH"]
        # Added the missing component display names
        component_display_names = {
            "C4B": "Alkyl Butane", "IS1": "Isomerate", "RFL": "Reformate",
            "F5X": "Mixed RFC", "RCG": "FCC Gasoline", "IC4": "DIB IC4",
            "HBY": "SHIP C4", "AKK": "Alkylate", "ETH": "Ethanol"
        }
        all_properties = ["SPG", "SUL", "RON", "MON", "RVP", "E70", "E10", "E15", "ARO", "BEN", "OXY", "OLEFIN"]

        for comp_html_key in component_html_keys:
            try:
                # Use a more robust .get() method to prevent KeyError and default to the key itself
                comp_tag = component_display_names.get(comp_html_key, comp_html_key)
                
                factor_str = request.form.get(f'component_{comp_html_key}_factor', '1.0').strip()
                factor = float(factor_str) if factor_str else 1.0
                calculated_cost = factor * regular_gasoline_price
                
                availability_str = request.form.get(f'component_{comp_html_key}_availability', '0').strip()
                availability = float(availability_str) if availability_str else 0.0
                
                min_comp_str = request.form.get(f'component_{comp_html_key}_min_comp', '0.000000').strip()
                min_comp = float(min_comp_str) if min_comp_str else 0.0
                
                comp_properties = {}
                for prop in all_properties:
                    prop_val_str = request.form.get(f'component_{comp_html_key}_property_{prop}', '0.0').strip()
                    try:
                        prop_val = float(prop_val_str or '0')
                    except ValueError:
                        prop_val = 0.0
                    comp_properties[prop] = prop_val
                components_data.append({
                    # Use the short key as the name for consistency
                    "name": comp_html_key,
                    "tag": comp_tag,
                    "cost": calculated_cost,
                    "availability": availability,
                    "min_comp": min_comp,
                    "factor": factor,
                    "properties": comp_properties
                })
                print(f"Component {comp_html_key}: availability={availability}, cost={calculated_cost}")
            except ValueError as e:
                error_msg = f"Invalid input for component {comp_tag}: {e}"
                print(f"ERROR: {error_msg}")
                return error_msg, 400

        specs_data = {}
        for prop in all_properties:
            specs_data[prop] = {}
            for grade in grades_data:
                try:
                    min_spec_str = request.form.get(f'spec_{prop}_{grade["name"]}_min', '0').strip()
                    max_spec_str = request.form.get(f'spec_{prop}_{grade["name"]}_max', 'inf').strip()
                    min_spec_val = float(min_spec_str) if min_spec_str and min_spec_str.lower() != 'inf' else 0.0
                    max_spec_val = float(max_spec_str) if max_spec_str and max_spec_str.lower() != 'inf' else float('inf')
                    specs_data[prop][grade['name']] = {"min": min_spec_val, "max": max_spec_val}
                except ValueError as e:
                    error_msg = f"Invalid input for spec {prop} for {grade['name']}: {e}"
                    print(f"ERROR: {error_msg}")
                    return error_msg, 400

        solver_choice = request.form.get('solver_choice', 'CBC')
        print(f"Using solver: {solver_choice}")

        internal_properties_list = ["SPG", "SUL", "RON", "ROI", "MON", "MOI", "RVP", "RVI", "E70", "E10", "E15", "ARO", "BEN", "OXY", "OLEFIN"]
        
        print("Starting optimization...")
        result1_content, result2_content, infeasibility_content = run_optimization(
            grades_data, components_data, internal_properties_list, specs_data, solver_choice
        )
        print("Optimization completed successfully")

        # Write results to files
        print(f"Writing result files to {BASE_PATH}...")
        with open(RESULT_FILE_NAME, "w", encoding="utf-8") as f1:
            f1.write(result1_content)
        with open(RANGE_REPORT_FILE_NAME, "w", encoding="utf-8") as f2:
            f2.write(result2_content)
        with open(INFEASIBILITY_FILE_NAME, "w", encoding="utf-8") as f3:
            f3.write(infeasibility_content)
        print("Files written successfully")

        return render_template('results.html', 
                                result1_filename=os.path.basename(RESULT_FILE_NAME), 
                                result2_filename=os.path.basename(RANGE_REPORT_FILE_NAME),
                                infeasibility_filename=os.path.basename(INFEASIBILITY_FILE_NAME))

    except Exception as e:
        print(f"🔥 CRITICAL ERROR in run_lp: {e}")
        traceback.print_exc()
        return f"INTERNAL ERROR: {str(e)}", 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        # Only allow downloading files we created
        allowed_files = [
            os.path.basename(RESULT_FILE_NAME),
            os.path.basename(RANGE_REPORT_FILE_NAME), 
            os.path.basename(INFEASIBILITY_FILE_NAME)
        ]
        
        if os.path.basename(filename) in allowed_files:
            full_path = os.path.join(BASE_PATH, os.path.basename(filename))
            if os.path.exists(full_path):
                return send_file(full_path, as_attachment=True)
            else:
                print(f"File not found: {full_path}")
                return "File not found.", 404
        else:
            print(f"Unauthorized file access attempt: {filename}")
            return "Unauthorized file access.", 403
            
    except Exception as e:
        print(f"Error in download_file: {e}")
        return f"Download error: {str(e)}", 500

# Health check endpoint for Render
@app.route('/health')
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}, 200

# Main application entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)




            
    
