import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
from io import BytesIO
from datetime import datetime

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Heat Exchanger FIV Analysis",
    page_icon="🔧",
    layout="wide"
)

# Constants (TEMA/ASME/ISO Standards)
GRAVITY = 9.81  # m/s²
SPEED_OF_SOUND_WATER = 1481  # m/s
FEI_CONSTANT = 3.0  # ASME Sec III Div 1 N-1321
STROUHAL_NUMBERS = {
    "Triangular": 0.33,  # TEMA RCB-4.521
    "Square": 0.21,
    "Rotated Square": 0.24,
    "Rotated Triangular": 0.35
}

# Security
API_KEY = "M_A_K_1995"
user_key = st.sidebar.text_input("Enter API Key:", type="password")

if user_key != API_KEY:
    st.error("⚠️ Unauthorized access. Please enter a valid API Key.")
    st.stop()

# Now your app can start
st.title("Heat Exchanger FIV Analysis")
st.write("Comprehensive Flow-Induced Vibration Analysis Tool for Shell-and-Tube Heat Exchangers")

# Sidebar Inputs
with st.sidebar:
    st.header("📌 Design Parameters (TEMA CEM Type)")
    
    # Tube parameters
    tube_od = st.number_input("Tube OD (mm)", min_value=5.0, max_value=50.0, value=19.5, step=0.1)
    tube_thickness = st.number_input("Tube thickness (mm)", min_value=0.1, max_value=5.0, value=1.27, step=0.01)
    tube_id = tube_od - 2 * tube_thickness
    st.text_input("Tube ID (mm)", value=f"{tube_id:.2f}", disabled=True)
    tube_length = st.number_input("Tube length (mm)", min_value=1000.0, max_value=10000.0, value=3580.0, step=10.0)
    density_tube_material = st.number_input("Density (kg/mm³)", min_value=1e-6, max_value=1e-4, value=8.03e-6, format="%.2e")
    permissible_stress = st.number_input("Permissible stress (N/mm²)", min_value=10.0, max_value=500.0, value=54.1, step=1.0)
    modulus_elasticity = st.number_input("Modulus (N/mm²)", min_value=1e3, max_value=3e5, value=1.95e5, step=1000.0)
    
    # Baffle parameters
    baffle_thickness = st.number_input("Baffle thickness (mm)", min_value=5.0, max_value=50.0, value=15.875, step=0.1)
    baffle_spacing_inlet = st.number_input("Inlet spacing (mm)", min_value=100.0, max_value=2000.0, value=1031.75, step=10.0)
    baffle_spacing_mid = st.number_input("Mid spacing (mm)", min_value=100.0, max_value=2000.0, value=470.0, step=10.0)
    baffle_spacing_outlet = st.number_input("Outlet spacing (mm)", min_value=100.0, max_value=2000.0, value=1031.75, step=10.0)
    
    # Fluid parameters
    shell_side_fluid_density = st.number_input("Shell density (kg/mm³)", min_value=1e-7, max_value=1e-4, value=1e-6, format="%.1e")
    tube_side_fluid_density = st.number_input("Tube density (kg/mm³)", min_value=1e-7, max_value=1e-4, value=1e-6, format="%.1e")
    flow_velocity = st.number_input("Flow velocity (m/s)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    
    # Layout parameters
    tube_pitch = st.number_input("Tube pitch (mm)", min_value=10.0, max_value=50.0, value=23.8125, step=0.1)
    diametral_clearance = st.number_input("Diametral clearance (mm)", min_value=0.1, max_value=2.0, value=0.49276, step=0.01)
    tube_array_pattern = st.selectbox("Tube pattern", list(STROUHAL_NUMBERS.keys()))
    damping_ratio = st.number_input("Damping ratio", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

def calculate_vibration_parameters(params):
    results = {}
    
    # Convert units to SI
    tube_od = params['tube_od'] / 1000  # mm to m
    tube_id = (params['tube_od'] - 2 * params['tube_thickness']) / 1000
    tube_length = params['tube_length'] / 1000
    tube_pitch = params['tube_pitch'] / 1000
    
    # Tube properties (ASME BPVC Section VIII Div 2)
    tube_cross_area = math.pi * (tube_od**2 - tube_id**2) / 4
    tube_mass_per_length = tube_cross_area * params['density_tube_material'] * 1e9
    tube_moment_inertia = math.pi * (tube_od**4 - tube_id**4) / 64
    
    # 1. Natural Frequency (ASME Sec III Div 1 N-1300)
    E = params['modulus_elasticity'] * 1e6  # N/mm² to N/m²
    results['Natural Frequency'] = (3.516 / (2 * math.pi)) * math.sqrt((E * tube_moment_inertia) / 
                                  (tube_mass_per_length * tube_length**4))
    
    # 2. Vortex Shedding (TEMA RCB-4.52)
    strouhal = STROUHAL_NUMBERS[params['tube_array_pattern']]
    results['Strouhal Number'] = strouhal
    results['Vortex Shedding Frequency'] = strouhal * params['flow_velocity'] / tube_od
    
    # 3. Turbulent Buffeting (TEMA RCB-4.53)
    results['Turbulent Buffeting Force'] = 0.5 * params['shell_side_fluid_density'] * 1e9 * \
                                         (params['flow_velocity']**2) * tube_od * tube_length
    
    # 4. Fluid Elastic Instability (ASME Sec III Div 1 N-1321)
    mass_damping = (2 * math.pi * params['damping_ratio'] * tube_mass_per_length) / \
                  (params['shell_side_fluid_density'] * 1e9 * tube_od**2)
    results['Fluid Elastic Instability Factor'] = FEI_CONSTANT * math.sqrt(mass_damping)
    results['Critical Reduced Velocity'] = results['Fluid Elastic Instability Factor'] * results['Natural Frequency'] * tube_od
    
    # 5. Acoustic Resonance (TEMA RCB-4.54)
    results['Axial Resonance'] = SPEED_OF_SOUND_WATER / (2 * tube_length)
    results['Angular Resonance'] = SPEED_OF_SOUND_WATER / (2 * tube_pitch)
    
    # 6. Mid-span Deflection (ISO 19904)
    results['Max Displacement'] = (5 * tube_mass_per_length * GRAVITY * tube_length**4) / \
                                 (384 * E * tube_moment_inertia) * 1000  # mm
    
    # 7. Wear Damage (ASME Sec III Div 1 N-1500)
    results['Wear Contact Events'] = int(1e6 * params['flow_velocity']**3 * (params['baffle_thickness']/1000))
    
    # 8. Fatigue Analysis (ASME BPVC Section VIII Div 2)
    dynamic_pressure = 0.5 * params['shell_side_fluid_density'] * 1e9 * params['flow_velocity']**2
    results['Fatigue Stress'] = dynamic_pressure * tube_od / (2 * params['tube_thickness']/1000) / 1e6  # MPa
    
    # 9. Noise Level (OSHA 1910.95)
    results['Noise Level'] = 20 * math.log10(params['flow_velocity'] * 100)  # dB
    
    # 10. Pressure Drop (TEMA Class R)
    results['Pressure Drop'] = 0.1 * (tube_length/(params['baffle_spacing_mid']/1000)) * \
                             params['shell_side_fluid_density'] * 1e9 * params['flow_velocity']**2 / 1e5  # bar
    
    return results

def check_acceptance_criteria(results, params):
    criteria = {}
    
    # 1. Vortex Shedding (TEMA RCB-4.521)
    ratio = results['Vortex Shedding Frequency'] / results['Natural Frequency']
    criteria['Vortex Shedding'] = {
        'Status': ratio < 0.5 or ratio > 1.5,
        'Value': f"{ratio:.2f}",
        'Limit': "0.5-1.5"
    }
    
    # 2. Turbulent Buffeting (TEMA RCB-4.531)
    criteria['Turbulent Buffeting'] = {
        'Status': results['Turbulent Buffeting Force'] < 1000,
        'Value': f"{results['Turbulent Buffeting Force']:.1f} N",
        'Limit': "<1000 N"
    }
    
    # 3. Fluid Elastic Instability (ASME N-1321)
    velocity_ratio = params['flow_velocity'] / results['Critical Reduced Velocity']
    criteria['Fluid Elastic Instability'] = {
        'Status': velocity_ratio < 0.5,
        'Value': f"{velocity_ratio:.2f}",
        'Limit': "<0.5"
    }
    
    # 4. Acoustic Resonance (TEMA RCB-4.541)
    axial_ratio = results['Vortex Shedding Frequency'] / results['Axial Resonance']
    angular_ratio = results['Vortex Shedding Frequency'] / results['Angular Resonance']
    criteria['Acoustic Resonance'] = {
        'Status': (axial_ratio < 0.8 or axial_ratio > 1.2) and (angular_ratio < 0.8 or angular_ratio > 1.2),
        'Value': f"Axial: {axial_ratio:.2f}, Angular: {angular_ratio:.2f}",
        'Limit': "0.8-1.2"
    }
    
    # 5. Mid-span Collision (ISO 19904)
    max_deflection = params['diametral_clearance'] / 2
    criteria['Mid-span Collision'] = {
        'Status': results['Max Displacement'] < max_deflection,
        'Value': f"{results['Max Displacement']:.2f} mm",
        'Limit': f"<{max_deflection:.2f} mm"
    }
    
    # 6. Wear Damage (ASME Sec III Div 1 N-1521)
    criteria['Wear Damage'] = {
        'Status': results['Wear Contact Events'] < 10000,
        'Value': f"{results['Wear Contact Events']}",
        'Limit': "<10000"
    }
    
    # 7. Fatigue Failure (ASME BPVC Section VIII Div 2)
    criteria['Fatigue Failure'] = {
        'Status': results['Fatigue Stress'] < 0.5*params['permissible_stress'],
        'Value': f"{results['Fatigue Stress']:.1f} MPa",
        'Limit': f"<{0.5*params['permissible_stress']:.1f} MPa"
    }
    
    # 8. Excessive Noise (OSHA 1910.95)
    criteria['Excessive Noise'] = {
        'Status': results['Noise Level'] < 85,
        'Value': f"{results['Noise Level']:.1f} dB",
        'Limit': "<85 dB"
    }
    
    # 9. Pressure Drop (TEMA Class R)
    criteria['Pressure Drop'] = {
        'Status': results['Pressure Drop'] < 1.0,
        'Value': f"{results['Pressure Drop']:.2f} bar",
        'Limit': "<1.0 bar"
    }
    
    # 10. Stress Corrosion (ASME Sec III Div 1 N-1331)
    criteria['Stress Corrosion'] = {
        'Status': results['Fatigue Stress'] < 0.3*params['permissible_stress'],
        'Value': f"{results['Fatigue Stress']:.1f} MPa",
        'Limit': f"<{0.3*params['permissible_stress']:.1f} MPa"
    }
    
    return criteria

def create_velocity_vibration_graph(results, params):
    """Create a graph showing vibration risk vs flow velocity"""
    velocities = np.linspace(0.1, 5.0, 50)  # Range of velocities from 0.1 to 5 m/s
    
    # Calculate vortex shedding frequencies across velocity range
    strouhal = STROUHAL_NUMBERS[params['tube_array_pattern']]
    vortex_freqs = [strouhal * v / (params['tube_od']/1000) for v in velocities]
    
    # Calculate FEI critical velocities
    mass_damping = (2 * math.pi * params['damping_ratio'] * 
                   (math.pi * ((params['tube_od']/1000)**2 - 
                    ((params['tube_od']-2*params['tube_thickness'])/1000)**2) / 4 * 
                    params['density_tube_material'] * 1e9)) / \
                  (params['shell_side_fluid_density'] * 1e9 * (params['tube_od']/1000)**2)
    fei_factor = FEI_CONSTANT * math.sqrt(mass_damping)
    critical_velocities = [fei_factor * results['Natural Frequency'] * (params['tube_od']/1000)] * len(velocities)
    
    # Calculate ratios
    vortex_ratios = [vf / results['Natural Frequency'] for vf in vortex_freqs]
    fei_ratios = [v / critical_velocities[0] for v in velocities]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(velocities, vortex_ratios, 'b-', label='Vortex Shedding Ratio (f/fn)')
    ax.plot(velocities, fei_ratios, 'r--', label='FEI Ratio (V/Vc)')
    
    # Add critical regions
    ax.axhspan(0.8, 1.2, color='red', alpha=0.1, label='Vortex Shedding Danger Zone')
    ax.axhline(0.5, color='green', linestyle=':', label='FEI Threshold')
    ax.axvline(params['flow_velocity'], color='black', linestyle='-', 
               label=f'Design Velocity ({params["flow_velocity"]} m/s)')
    
    ax.set_title('Vibration Risk vs Flow Velocity')
    ax.set_xlabel('Flow Velocity (m/s)')
    ax.set_ylabel('Vibration Risk Ratio')
    ax.grid(True)
    ax.legend()
    
    return fig

def create_vibration_graph(results, params):
    time = np.linspace(0, 1, 1000)
    displacement = results['Max Displacement'] * np.sin(2 * np.pi * results['Natural Frequency'] * time)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, displacement, 'b-', linewidth=2)
    ax.set_title('Tube Vibration Response')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement (mm)')
    ax.grid(True)
    return fig

def add_diagonal_watermark(fig):
    """Add diagonal watermark to a figure"""
    fig.text(0.5, 0.5, 'Manoj Abhay Kulkarni',
             rotation=45, fontsize=40, color='gray',
             alpha=0.1, ha='center', va='center',
             bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.1))

def create_pdf_report(params, results, criteria):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.pyplot import figure, text, axis, savefig, close
    from matplotlib.table import Table
    import numpy as np
    
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Title Page
        fig = figure(figsize=(8.5, 11))
        text(0.5, 0.8, 'CEM Heat Exchanger FIV Analysis Report', 
             ha='center', va='center', fontsize=18, fontweight='bold')
        text(0.5, 0.75, 'TEMA/ASME/ISO Standards Compliance', 
             ha='center', va='center', fontsize=14)
        text(0.5, 0.7, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
             ha='center', va='center', fontsize=12)
        
        # Add diagonal watermark
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Table of Contents
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.9, 'Table of Contents', fontsize=16, fontweight='bold')
        text(0.1, 0.8, '1. Executive Summary', fontsize=12)
        text(0.1, 0.75, '2. Vibration Mechanism Analysis', fontsize=12)
        text(0.1, 0.7, '3. Damage Potential Assessment', fontsize=12)
        text(0.1, 0.65, '4. Visualizations', fontsize=12)
        text(0.1, 0.6, '5. Detailed Acceptance Criteria', fontsize=12)
        text(0.1, 0.55, '6. Design Parameters', fontsize=12)
        
        # Add diagonal watermark
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Executive Summary
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.9, '1. Executive Summary', fontsize=16, fontweight='bold')
        
        # Overall status
        all_acceptable = all([criteria[key]['Status'] for key in criteria])
        status_text = "DESIGN IS ACCEPTABLE" if all_acceptable else "DESIGN REQUIRES REVIEW"
        status_color = 'green' if all_acceptable else 'red'
        text(0.1, 0.85, f"Overall Status: {status_text}", 
             fontsize=14, color=status_color, fontweight='bold')
        
        # Key findings
        text(0.1, 0.8, "Key Findings:", fontsize=12, fontweight='bold')
        
        # Find the most critical issue if any
        critical_issues = [key for key in criteria if not criteria[key]['Status']]
        if critical_issues:
            text(0.1, 0.75, f"Critical Issues Found: {len(critical_issues)}", 
                 fontsize=12, color='red')
            for i, issue in enumerate(critical_issues[:3]):  # Show top 3 issues
                text(0.15, 0.7 - i*0.05, f"- {issue}", fontsize=10)
        else:
            text(0.1, 0.75, "No critical issues found", fontsize=12, color='green')
        
        # Recommendations
        text(0.1, 0.55, "Recommendations:", fontsize=12, fontweight='bold')
        if all_acceptable:
            text(0.15, 0.5, "- Proceed with current design", fontsize=10)
        else:
            text(0.15, 0.5, "- Review critical vibration mechanisms", fontsize=10)
            text(0.15, 0.45, "- Consider design modifications", fontsize=10)
            text(0.15, 0.4, "- Perform detailed stress analysis if needed", fontsize=10)
        
        # Add diagonal watermark
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Vibration Mechanism Analysis
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, '2. Vibration Mechanism Analysis', fontsize=16, fontweight='bold')
        
        # Vortex Shedding
        text(0.1, 0.9, 'VORTEX SHEDDING', fontsize=12, fontweight='bold')
        text(0.1, 0.87, f"Natural Frequency: {results['Natural Frequency']:.2f} Hz", fontsize=10)
        text(0.1, 0.84, f"Strouhal Number: {results['Strouhal Number']:.2f}", fontsize=10)
        text(0.1, 0.81, f"Vortex Shedding Frequency: {results['Vortex Shedding Frequency']:.2f} Hz", fontsize=10)
        
        vs_status = criteria['Vortex Shedding']['Status']
        vs_color = 'green' if vs_status else 'red'
        text(0.1, 0.78, f"Status: {'ACCEPTABLE' if vs_status else 'NOT ACCEPTABLE'}", 
             fontsize=10, color=vs_color, fontweight='bold')
        
        # Turbulent Buffeting
        text(0.1, 0.73, 'TURBULENT BUFFETING', fontsize=12, fontweight='bold')
        text(0.1, 0.7, f"Buffeting Force: {results['Turbulent Buffeting Force']:.1f} N", fontsize=10)
        
        tb_status = criteria['Turbulent Buffeting']['Status']
        tb_color = 'green' if tb_status else 'red'
        text(0.1, 0.67, f"Status: {'ACCEPTABLE' if tb_status else 'NOT ACCEPTABLE'}", 
             fontsize=10, color=tb_color, fontweight='bold')
        
        # Fluid Elastic Instability
        text(0.55, 0.9, 'FLUID ELASTIC INSTABILITY', fontsize=12, fontweight='bold')
        text(0.55, 0.87, f"Instability Factor: {results['Fluid Elastic Instability Factor']:.2f}", fontsize=10)
        text(0.55, 0.84, f"Critical Velocity: {results['Critical Reduced Velocity']:.2f} m/s", fontsize=10)
        
        fei_status = criteria['Fluid Elastic Instability']['Status']
        fei_color = 'green' if fei_status else 'red'
        text(0.55, 0.81, f"Status: {'ACCEPTABLE' if fei_status else 'NOT ACCEPTABLE'}", 
             fontsize=10, color=fei_color, fontweight='bold')
        
        # Acoustic Resonance
        text(0.55, 0.73, 'ACOUSTIC RESONANCE', fontsize=12, fontweight='bold')
        text(0.55, 0.7, f"Axial Resonance: {results['Axial Resonance']:.2f} Hz", fontsize=10)
        text(0.55, 0.67, f"Angular Resonance: {results['Angular Resonance']:.2f} Hz", fontsize=10)
        
        ar_status = criteria['Acoustic Resonance']['Status']
        ar_color = 'green' if ar_status else 'red'
        text(0.55, 0.64, f"Status: {'ACCEPTABLE' if ar_status else 'NOT ACCEPTABLE'}", 
             fontsize=10, color=ar_color, fontweight='bold')
        
        # Add diagonal watermark
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Damage Potential Assessment
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, '3. Damage Potential Assessment', fontsize=16, fontweight='bold')
        
        # Column 1
        text(0.1, 0.9, 'DISPLACEMENT & COLLISION', fontsize=12, fontweight='bold')
        text(0.1, 0.87, f"Max Displacement: {results['Max Displacement']:.2f} mm", fontsize=10)
        
        mc_status = criteria['Mid-span Collision']['Status']
        mc_color = 'green' if mc_status else 'red'
        text(0.1, 0.84, f"Collision Risk: {'NO' if mc_status else 'YES'}", 
             fontsize=10, color=mc_color)
        
        # Column 2
        text(0.4, 0.9, 'WEAR & FATIGUE', fontsize=12, fontweight='bold')
        text(0.4, 0.87, f"Wear Events: {results['Wear Contact Events']}", fontsize=10)
        
        wd_status = criteria['Wear Damage']['Status']
        wd_color = 'green' if wd_status else 'red'
        text(0.4, 0.84, f"Status: {'ACCEPTABLE' if wd_status else 'EXCESSIVE'}", 
             fontsize=10, color=wd_color)
        
        # Column 3
        text(0.7, 0.9, 'OTHER FACTORS', fontsize=12, fontweight='bold')
        text(0.7, 0.87, f"Noise Level: {results['Noise Level']:.1f} dB", fontsize=10)
        
        sc_status = criteria['Stress Corrosion']['Status']
        sc_color = 'green' if sc_status else 'red'
        text(0.7, 0.84, f"Corrosion Risk: {'LOW' if sc_status else 'HIGH'}", 
             fontsize=10, color=sc_color)
        
        # Add a table for all damage factors
        fig_table = figure(figsize=(8, 4))
        ax = fig_table.add_subplot(111)
        
        damage_data = [
            ["Max Displacement", f"{results['Max Displacement']:.2f} mm", 
             f"<{params['diametral_clearance']/2:.2f} mm", 
             "PASS" if criteria['Mid-span Collision']['Status'] else "FAIL"],
            ["Wear Events", f"{results['Wear Contact Events']}", "<10000", 
             "PASS" if criteria['Wear Damage']['Status'] else "FAIL"],
            ["Fatigue Stress", f"{results['Fatigue Stress']:.1f} MPa", 
             f"<{0.5*params['permissible_stress']:.1f} MPa", 
             "PASS" if criteria['Fatigue Failure']['Status'] else "FAIL"],
            ["Corrosion Risk", "LOW" if criteria['Stress Corrosion']['Status'] else "HIGH", 
             "LOW", 
             "PASS" if criteria['Stress Corrosion']['Status'] else "FAIL"]
        ]
        
        table = ax.table(cellText=damage_data,
                        colLabels=["Parameter", "Value", "Limit", "Status"],
                        loc='center',
                        cellLoc='center')
        
        # Color cells based on status
        for i in range(1, len(damage_data)+1):
            status_cell = table[i, 3]
            if "FAIL" in damage_data[i-1][3]:
                status_cell.set_facecolor('lightcoral')
            else:
                status_cell.set_facecolor('lightgreen')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.axis('off')
        ax.set_title('Damage Potential Summary', fontweight='bold')
        
        # Add diagonal watermark to table figure
        add_diagonal_watermark(fig_table)
        
        pdf.savefig(fig_table, bbox_inches='tight')
        close(fig_table)
        
        # Add diagonal watermark to main figure
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Visualizations
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, '4. Visualizations', fontsize=16, fontweight='bold')
        axis('off')
        
        # Add diagonal watermark
        add_diagonal_watermark(fig)
        
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Add velocity-vibration graph
        fig_vel = create_velocity_vibration_graph(results, params)
        add_diagonal_watermark(fig_vel)
        pdf.savefig(fig_vel, bbox_inches='tight')
        close(fig_vel)
        
        # Add vibration graph
        fig_vib = create_vibration_graph(results, params)
        add_diagonal_watermark(fig_vib)
        pdf.savefig(fig_vib, bbox_inches='tight')
        close(fig_vib)
        
        # Detailed Acceptance Criteria
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, '5. Detailed Acceptance Criteria', fontsize=16, fontweight='bold')
        axis('off')
        
        # Create a table with all criteria
        fig_table = figure(figsize=(8, 8))
        ax = fig_table.add_subplot(111)
        
        criteria_data = []
        for key in criteria:
            criteria_data.append([
                key,
                criteria[key]['Value'],
                criteria[key]['Limit'],
                "PASS" if criteria[key]['Status'] else "FAIL"
            ])
        
        table = ax.table(cellText=criteria_data,
                        colLabels=["Mechanism", "Value", "Limit", "Status"],
                        loc='center',
                        cellLoc='center')
        
        # Color cells based on status
        for i in range(1, len(criteria_data)+1):
            status_cell = table[i, 3]
            if "FAIL" in criteria_data[i-1][3]:
                status_cell.set_facecolor('lightcoral')
            else:
                status_cell.set_facecolor('lightgreen')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax.axis('off')
        ax.set_title('Detailed Acceptance Criteria', fontweight='bold')
        
        # Add diagonal watermark to table figure
        add_diagonal_watermark(fig_table)
        
        pdf.savefig(fig_table, bbox_inches='tight')
        close(fig_table)
        
        # Add diagonal watermark to main figure
        add_diagonal_watermark(fig)
        
        pdf.savefig(fig, bbox_inches='tight')
        close()
        
        # Design Parameters
        fig = figure(figsize=(8.5, 11))
        text(0.1, 0.95, '6. Design Parameters', fontsize=16, fontweight='bold')
        
        # Tube parameters
        text(0.1, 0.9, 'Tube Parameters:', fontsize=12, fontweight='bold')
        text(0.1, 0.87, f"OD: {params['tube_od']} mm", fontsize=10)
        text(0.1, 0.84, f"Thickness: {params['tube_thickness']} mm", fontsize=10)
        text(0.1, 0.81, f"Length: {params['tube_length']} mm", fontsize=10)
        text(0.1, 0.78, f"Material Density: {params['density_tube_material']:.2e} kg/mm³", fontsize=10)
        
        # Baffle parameters
        text(0.5, 0.9, 'Baffle Parameters:', fontsize=12, fontweight='bold')
        text(0.5, 0.87, f"Thickness: {params['baffle_thickness']} mm", fontsize=10)
        text(0.5, 0.84, f"Inlet Spacing: {params['baffle_spacing_inlet']} mm", fontsize=10)
        text(0.5, 0.81, f"Mid Spacing: {params['baffle_spacing_mid']} mm", fontsize=10)
        text(0.5, 0.78, f"Outlet Spacing: {params['baffle_spacing_outlet']} mm", fontsize=10)
        
        # Fluid parameters
        text(0.1, 0.7, 'Fluid Parameters:', fontsize=12, fontweight='bold')
        text(0.1, 0.67, f"Shell Density: {params['shell_side_fluid_density']:.1e} kg/mm³", fontsize=10)
        text(0.1, 0.64, f"Tube Density: {params['tube_side_fluid_density']:.1e} kg/mm³", fontsize=10)
        text(0.1, 0.61, f"Flow Velocity: {params['flow_velocity']} m/s", fontsize=10)
        
        # Layout parameters
        text(0.5, 0.7, 'Layout Parameters:', fontsize=12, fontweight='bold')
        text(0.5, 0.67, f"Tube Pitch: {params['tube_pitch']} mm", fontsize=10)
        text(0.5, 0.64, f"Clearance: {params['diametral_clearance']} mm", fontsize=10)
        text(0.5, 0.61, f"Pattern: {params['tube_array_pattern']}", fontsize=10)
        text(0.5, 0.58, f"Damping Ratio: {params['damping_ratio']}", fontsize=10)
        
        # Add diagonal watermark
        add_diagonal_watermark(fig)
        
        axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        close()
    
    return buf

# Main App
st.title("CEM Heat Exchanger Flow-Induced Vibration Analysis")
st.subheader("TEMA/ASME/ISO Standards Compliance")

# Prepare parameters
params = {
    'tube_od': tube_od,
    'tube_thickness': tube_thickness,
    'tube_length': tube_length,
    'density_tube_material': density_tube_material,
    'permissible_stress': permissible_stress,
    'modulus_elasticity': modulus_elasticity,
    'baffle_thickness': baffle_thickness,
    'shell_side_fluid_density': shell_side_fluid_density,
    'tube_side_fluid_density': tube_side_fluid_density,
    'baffle_spacing_inlet': baffle_spacing_inlet,
    'baffle_spacing_mid': baffle_spacing_mid,
    'baffle_spacing_outlet': baffle_spacing_outlet,
    'flow_velocity': flow_velocity,
    'tube_pitch': tube_pitch,
    'diametral_clearance': diametral_clearance,
    'tube_array_pattern': tube_array_pattern,
    'damping_ratio': damping_ratio
}

# Calculations
results = calculate_vibration_parameters(params)
criteria = check_acceptance_criteria(results, params)

# Display Results in Output Summary Format
st.header("OUTPUT SUMMARY")
st.subheader("FLOW INDUCED VIBRATION MECHANISMS")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### VORTEX SHEDDING")
    st.write(f"I) Natural Frequency = {results['Natural Frequency']:.2f} Hz")
    st.write(f"II) Strouhal number = {results['Strouhal Number']:.2f}")
    st.write(f"III) Vortex shedding frequency = {results['Vortex Shedding Frequency']:.2f} Hz")
    status = "✅ ACCEPTABLE" if criteria['Vortex Shedding']['Status'] else "❌ NOT ACCEPTABLE"
    st.markdown(f"**STATUS OF VORTEX SHEDDING:** {status}")

    st.markdown("### TURBULENT BUFFETING")
    st.write(f"I) Turbulent Buffeting Force = {results['Turbulent Buffeting Force']:.1f} N")
    status = "✅ ACCEPTABLE" if criteria['Turbulent Buffeting']['Status'] else "❌ NOT ACCEPTABLE"
    st.markdown(f"**STATUS OF TURBULENT BUFFETING:** {status}")

with col2:
    st.markdown("### FLUID ELASTIC INSTABILITY")
    st.write(f"I) Fluid Elastic Instability Factor = {results['Fluid Elastic Instability Factor']:.2f}")
    st.write(f"II) Critical Reduced Velocity = {results['Critical Reduced Velocity']:.2f} m/s")
    status = "✅ ACCEPTABLE" if criteria['Fluid Elastic Instability']['Status'] else "❌ NOT ACCEPTABLE"
    st.markdown(f"**STATUS OF FLUID ELASTIC INSTABILITY:** {status}")

    st.markdown("### ACOUSTIC RESONANCE")
    st.write(f"Axial Resonance = {results['Axial Resonance']:.2f} Hz")
    st.write(f"Angular Resonance = {results['Angular Resonance']:.2f} Hz")
    status = "✅ ACCEPTABLE" if criteria['Acoustic Resonance']['Status'] else "❌ NOT ACCEPTABLE"
    st.markdown(f"**STATUS OF ACOUSTIC RESONANCE:** {status}")

# Damage Effects
st.subheader("POSSIBILITY DAMAGING EFFECT OF THE FIV ON HEAT EXCHANGER")
damage_cols = st.columns(3)
with damage_cols[0]:
    st.write(f"I) Max Displacement = {results['Max Displacement']:.2f} mm")
    st.write(f"IV) Noise Level = {results['Noise Level']:.1f} dB")
with damage_cols[1]:
    st.write(f"II) Mid-span Collision Risk = {'YES' if not criteria['Mid-span Collision']['Status'] else 'NO'}")
    st.write(f"V) Pressure Drop = {results['Pressure Drop']:.2f} bar")
with damage_cols[2]:
    st.write(f"III) Wear Contact Events = {results['Wear Contact Events']}")
    st.write(f"VI) Stress Corrosion Cracking Risk = {'HIGH' if not criteria['Stress Corrosion']['Status'] else 'LOW'}")

# Visualizations
st.header("Vibration Analysis Visualizations")
fig1 = create_velocity_vibration_graph(results, params)
fig2 = create_vibration_graph(results, params)

viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    st.pyplot(fig1)
    st.caption("Figure 1: Vibration risk factors vs flow velocity. Shows critical regions for vortex shedding and FEI.")
with viz_col2:
    st.pyplot(fig2)
    st.caption("Figure 2: Time-domain vibration response at natural frequency.")

# PDF Report Generation
if st.button("📥 Generate Comprehensive PDF Report"):
    with st.spinner('Generating professional report...'):
        pdf_buffer = create_pdf_report(params, results, criteria)
        
    st.success('Report generated successfully!')
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="CEM_Heat_Exchanger_FIV_Analysis.pdf",
        mime="application/pdf"
    )