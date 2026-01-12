# ============================================================================
# TASK B: FLEET ARBITRAGE DASHBOARD - COLAB VERSION
# ============================================================================

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Create sample compliance data
def create_sample_data():
    """Create sample compliance data for the dashboard"""

    vessels = [
        {
            "Vessel_ID": "NG001",
            "Ship_Type": "Container Ship",
            "Avg_GHG_Intensity": 0.045672,
            "Compliance_Status": "Deficit",
            "Total_Deficit": 15.245,
            "Total_Surplus": 0.0,
            "Total_CO2_kg": 1250000.5,
            "Total_Distance_NM": 25000.0,
            "Total_Fuel_MT": 352.2,
            "Avg_Engine_Efficiency": 78.5,
            "penalty_risk": "High"
        },
        {
            "Vessel_ID": "NG002",
            "Ship_Type": "Tanker",
            "Avg_GHG_Intensity": 0.038921,
            "Compliance_Status": "Surplus",
            "Total_Deficit": 0.0,
            "Total_Surplus": 8.178,
            "Total_CO2_kg": 980000.3,
            "Total_Distance_NM": 32000.0,
            "Total_Fuel_MT": 287.7,
            "Avg_Engine_Efficiency": 85.2,
            "penalty_risk": "None"
        },
        {
            "Vessel_ID": "NG003",
            "Ship_Type": "Bulk Carrier",
            "Avg_GHG_Intensity": 0.052341,
            "Compliance_Status": "Deficit",
            "Total_Deficit": 22.567,
            "Total_Surplus": 0.0,
            "Total_CO2_kg": 1560000.8,
            "Total_Distance_NM": 28000.0,
            "Total_Fuel_MT": 423.3,
            "Avg_Engine_Efficiency": 72.1,
            "penalty_risk": "High"
        },
        {
            "Vessel_ID": "NG004",
            "Ship_Type": "General Cargo",
            "Avg_GHG_Intensity": 0.036789,
            "Compliance_Status": "Surplus",
            "Total_Deficit": 0.0,
            "Total_Surplus": 12.543,
            "Total_CO2_kg": 890000.2,
            "Total_Distance_NM": 31000.0,
            "Total_Fuel_MT": 258.8,
            "Avg_Engine_Efficiency": 88.4,
            "penalty_risk": "None"
        },
        {
            "Vessel_ID": "NG005",
            "Ship_Type": "LNG Carrier",
            "Avg_GHG_Intensity": 0.032145,
            "Compliance_Status": "Surplus",
            "Total_Deficit": 0.0,
            "Total_Surplus": 18.321,
            "Total_CO2_kg": 750000.1,
            "Total_Distance_NM": 35000.0,
            "Total_Fuel_MT": 215.4,
            "Avg_Engine_Efficiency": 91.3,
            "penalty_risk": "None"
        },
        {
            "Vessel_ID": "NG006",
            "Ship_Type": "Oil Tanker",
            "Avg_GHG_Intensity": 0.048912,
            "Compliance_Status": "Deficit",
            "Total_Deficit": 10.891,
            "Total_Surplus": 0.0,
            "Total_CO2_kg": 1450000.0,
            "Total_Distance_NM": 29000.0,
            "Total_Fuel_MT": 389.9,
            "Avg_Engine_Efficiency": 75.3,
            "penalty_risk": "Medium"
        },
        {
            "Vessel_ID": "NG007",
            "Ship_Type": "Container Ship",
            "Avg_GHG_Intensity": 0.041234,
            "Compliance_Status": "Deficit",
            "Total_Deficit": 5.432,
            "Total_Surplus": 0.0,
            "Total_CO2_kg": 1100000.7,
            "Total_Distance_NM": 33000.0,
            "Total_Fuel_MT": 298.5,
            "Avg_Engine_Efficiency": 81.2,
            "penalty_risk": "Low"
        },
        {
            "Vessel_ID": "NG008",
            "Ship_Type": "Bulk Carrier",
            "Avg_GHG_Intensity": 0.039876,
            "Compliance_Status": "Surplus",
            "Total_Deficit": 0.0,
            "Total_Surplus": 6.789,
            "Total_CO2_kg": 950000.4,
            "Total_Distance_NM": 30000.0,
            "Total_Fuel_MT": 267.3,
            "Avg_Engine_Efficiency": 84.7,
            "penalty_risk": "None"
        }
    ]

    # Calculate metrics
    total_vessels = len(vessels)
    surplus_count = len([v for v in vessels if v["Compliance_Status"] == "Surplus"])
    deficit_count = len([v for v in vessels if v["Compliance_Status"] == "Deficit"])
    total_surplus = sum(v["Total_Surplus"] for v in vessels)
    total_deficit = sum(v["Total_Deficit"] for v in vessels)
    net_balance = total_surplus - total_deficit

    avg_intensity = sum(v["Avg_GHG_Intensity"] for v in vessels) / total_vessels

    metrics = {
        "total_vessels": total_vessels,
        "surplus_count": surplus_count,
        "deficit_count": deficit_count,
        "total_surplus": total_surplus,
        "total_deficit": total_deficit,
        "net_balance": net_balance,
        "compliance_rate": (surplus_count / total_vessels) * 100,
        "avg_intensity": avg_intensity,
        "target_intensity": avg_intensity * 0.95,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return pd.DataFrame(vessels), metrics

# Load data
vessels_df, metrics = create_sample_data()
print("✅ Sample compliance data created!")
print(f"   Total vessels: {metrics['total_vessels']}")
print(f"   Surplus vessels: {metrics['surplus_count']}")
print(f"   Deficit vessels: {metrics['deficit_count']}")
print(f"   Net balance: {metrics['net_balance']:.3f}")

# ============================================================================
# INTERACTIVE DASHBOARD USING PLOTLY
# ============================================================================

import plotly.subplots as sp
from ipywidgets import interact, interactive, fixed, widgets
from IPython.display import display, HTML, clear_output

class FleetArbitrageDashboard:
    """Interactive dashboard for Task B"""

    def __init__(self, vessels_df, metrics):
        self.vessels_df = vessels_df
        self.metrics = metrics
        self.selected_deficit = None
        self.selected_surplus = None
        self.pooling_result = None

        # Color maps
        self.status_colors = {
            "Surplus": "#10b981",  # Green
            "Deficit": "#ef4444"   # Red
        }

        self.risk_colors = {
            "High": "#ef4444",     # Red
            "Medium": "#f59e0b",   # Yellow
            "Low": "#10b981",      # Green
            "None": "#3b82f6"      # Blue
        }

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        """Create interactive widgets"""

        # Filter widgets
        self.status_filter = widgets.Dropdown(
            options=['All', 'Surplus', 'Deficit'],
            value='All',
            description='Filter by Status:',
            style={'description_width': 'initial'}
        )

        self.risk_filter = widgets.SelectMultiple(
            options=['High', 'Medium', 'Low', 'None'],
            value=['High', 'Medium', 'Low', 'None'],
            description='Filter by Risk:',
            rows=4,
            style={'description_width': 'initial'}
        )

        # Pooling simulator widgets
        deficit_options = self.vessels_df[self.vessels_df['Compliance_Status'] == 'Deficit']['Vessel_ID'].tolist()
        surplus_options = self.vessels_df[self.vessels_df['Compliance_Status'] == 'Surplus']['Vessel_ID'].tolist()

        self.deficit_select = widgets.Dropdown(
            options=[''] + deficit_options,
            value='',
            description='Deficit Vessel:',
            style={'description_width': 'initial'}
        )

        self.surplus_select = widgets.Dropdown(
            options=[''] + surplus_options,
            value='',
            description='Surplus Vessel:',
            style={'description_width': 'initial'}
        )

        self.simulate_btn = widgets.Button(
            description='🔄 Simulate Pooling',
            button_style='primary',
            layout=widgets.Layout(width='200px')
        )

        self.refresh_btn = widgets.Button(
            description='🔄 Refresh',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )

        # Connect events
        self.simulate_btn.on_click(self.simulate_pooling)
        self.refresh_btn.on_click(self.refresh_dashboard)

    def simulate_pooling(self, btn):
        """Simulate pooling between selected vessels"""
        if not self.deficit_select.value or not self.surplus_select.value:
            print("⚠️ Please select both deficit and surplus vessels")
            return

        deficit_vessel = self.vessels_df[self.vessels_df['Vessel_ID'] == self.deficit_select.value].iloc[0]
        surplus_vessel = self.vessels_df[self.vessels_df['Vessel_ID'] == self.surplus_select.value].iloc[0]

        deficit_amount = deficit_vessel['Total_Deficit']
        surplus_amount = surplus_vessel['Total_Surplus']

        # Calculate pooling
        credit_transfer = min(deficit_amount, surplus_amount)
        remaining_deficit = deficit_amount - credit_transfer
        remaining_surplus = surplus_amount - credit_transfer
        estimated_cost = credit_transfer * 200  # $200 per credit

        # Store result
        self.pooling_result = {
            'deficit_vessel': deficit_vessel['Vessel_ID'],
            'surplus_vessel': surplus_vessel['Vessel_ID'],
            'deficit_type': deficit_vessel['Ship_Type'],
            'surplus_type': surplus_vessel['Ship_Type'],
            'credit_transfer': credit_transfer,
            'remaining_deficit': remaining_deficit,
            'remaining_surplus': remaining_surplus,
            'estimated_cost': estimated_cost,
            'fully_covered': remaining_deficit <= 0
        }

        # Redraw dashboard
        self.display_dashboard()

    def refresh_dashboard(self, btn):
        """Refresh the dashboard"""
        self.pooling_result = None
        self.deficit_select.value = ''
        self.surplus_select.value = ''
        self.display_dashboard()

    def create_stats_cards(self):
        """Create statistics cards"""
        cards_html = f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Total Vessels</div>
                <div style="font-size: 28px; font-weight: bold; color: #1f2937;">{self.metrics['total_vessels']}</div>
            </div>

            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Surplus Vessels</div>
                <div style="font-size: 28px; font-weight: bold; color: #10b981;">{self.metrics['surplus_count']}</div>
            </div>

            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Deficit Vessels</div>
                <div style="font-size: 28px; font-weight: bold; color: #ef4444;">{self.metrics['deficit_count']}</div>
            </div>

            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Net Balance</div>
                <div style="font-size: 28px; font-weight: bold; color: { '#10b981' if self.metrics['net_balance'] >= 0 else '#ef4444' };">
                    {self.metrics['net_balance']:.3f}
                </div>
            </div>
        </div>
        """
        return cards_html

    def create_compliance_chart(self):
        """Create compliance distribution chart"""
        fig = go.Figure(data=[
            go.Pie(
                labels=['Surplus Vessels', 'Deficit Vessels'],
                values=[self.metrics['surplus_count'], self.metrics['deficit_count']],
                marker_colors=['#10b981', '#ef4444'],
                hole=.3
            )
        ])

        fig.update_layout(
            title_text="Compliance Distribution",
            height=300,
            showlegend=True,
            margin=dict(t=50, b=20, l=20, r=20)
        )

        return fig

    def create_intensity_chart(self):
        """Create GHG intensity comparison chart"""
        surplus_avg = self.vessels_df[self.vessels_df['Compliance_Status'] == 'Surplus']['Avg_GHG_Intensity'].mean()
        deficit_avg = self.vessels_df[self.vessels_df['Compliance_Status'] == 'Deficit']['Avg_GHG_Intensity'].mean()

        fig = go.Figure(data=[
            go.Bar(
                name='Surplus Avg',
                x=['Surplus Avg'],
                y=[surplus_avg],
                marker_color='#10b981'
            ),
            go.Bar(
                name='Deficit Avg',
                x=['Deficit Avg'],
                y=[deficit_avg],
                marker_color='#ef4444'
            ),
            go.Bar(
                name='Target',
                x=['Target'],
                y=[self.metrics['target_intensity']],
                marker_color='#3b82f6'
            )
        ])

        fig.update_layout(
            title_text="GHG Intensity Comparison",
            yaxis_title="kg CO2/NM-tonne",
            height=300,
            showlegend=True,
            margin=dict(t=50, b=40, l=60, r=20)
        )

        return fig

    def create_liability_table(self):
        """Create vessel liability table"""
        # Apply filters
        filtered_df = self.vessels_df.copy()

        if self.status_filter.value != 'All':
            filtered_df = filtered_df[filtered_df['Compliance_Status'] == self.status_filter.value]

        if self.risk_filter.value:
            filtered_df = filtered_df[filtered_df['penalty_risk'].isin(self.risk_filter.value)]

        # Create HTML table
        table_html = """
        <style>
            .vessel-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            .vessel-table th {
                background-color: #f9fafb;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                color: #374151;
                border-bottom: 2px solid #e5e7eb;
            }
            .vessel-table td {
                padding: 12px;
                border-bottom: 1px solid #e5e7eb;
            }
            .vessel-table tr:hover {
                background-color: #f9fafb;
            }
            .status-badge {
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
            }
            .status-surplus { background-color: #d1fae5; color: #065f46; }
            .status-deficit { background-color: #fee2e2; color: #991b1b; }
            .risk-badge {
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
            }
            .risk-high { background-color: #fee2e2; color: #991b1b; }
            .risk-medium { background-color: #fef3c7; color: #92400e; }
            .risk-low { background-color: #d1fae5; color: #065f46; }
            .risk-none { background-color: #dbeafe; color: #1e40af; }
        </style>

        <table class="vessel-table">
            <thead>
                <tr>
                    <th>Vessel ID</th>
                    <th>Ship Type</th>
                    <th>Status</th>
                    <th>GHG Intensity</th>
                    <th>Credits</th>
                    <th>Penalty Risk</th>
                    <th>Select</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, row in filtered_df.iterrows():
            status_class = "status-surplus" if row['Compliance_Status'] == 'Surplus' else "status-deficit"
            risk_class = f"risk-{row['penalty_risk'].lower()}"

            credits = f"+{row['Total_Surplus']:.3f}" if row['Compliance_Status'] == 'Surplus' else f"-{row['Total_Deficit']:.3f}"
            credits_color = "color: #10b981;" if row['Compliance_Status'] == 'Surplus' else "color: #ef4444;"

            table_html += f"""
                <tr>
                    <td><strong>{row['Vessel_ID']}</strong></td>
                    <td>{row['Ship_Type']}</td>
                    <td><span class="status-badge {status_class}">{row['Compliance_Status']}</span></td>
                    <td>{row['Avg_GHG_Intensity']:.6f}</td>
                    <td><strong style="{credits_color}">{credits}</strong></td>
                    <td><span class="risk-badge {risk_class}">{row['penalty_risk']}</span></td>
                    <td>
                        <button onclick="
                            document.getElementById('select-{row['Vessel_ID']}').click();
                            document.querySelector('[value=\"{row['Vessel_ID']}\"]').selected = true;
                        " style="
                            background: #3b82f6;
                            color: white;
                            border: none;
                            padding: 4px 12px;
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 12px;
                        ">
                            Select
                        </button>
                        <select id="select-{row['Vessel_ID']}" style="display: none;">
                            <option value="{row['Vessel_ID']}"></option>
                        </select>
                    </td>
                </tr>
            """

        table_html += """
            </tbody>
        </table>
        """

        return table_html

    def create_pooling_results(self):
        """Create pooling results display"""
        if not self.pooling_result:
            return ""

        result = self.pooling_result
        status_icon = "✅" if result['fully_covered'] else "⚠️"
        status_class = "background-color: #d1fae5; border-color: #10b981;" if result['fully_covered'] else "background-color: #fef3c7; border-color: #f59e0b;"

        result_html = f"""
        <div style="{status_class} border: 2px solid; border-radius: 10px; padding: 20px; margin-top: 20px;">
            <h3 style="margin-top: 0; color: {'#065f46' if result['fully_covered'] else '#92400e'}">
                {status_icon} Pooling Simulation Results
            </h3>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                <div>
                    <div style="font-size: 12px; color: #6b7280;">Deficit Vessel</div>
                    <div style="font-size: 16px; font-weight: bold;">{result['deficit_vessel']}</div>
                    <div style="font-size: 12px; color: #6b7280;">{result['deficit_type']}</div>
                </div>

                <div>
                    <div style="font-size: 12px; color: #6b7280;">Surplus Vessel</div>
                    <div style="font-size: 16px; font-weight: bold;">{result['surplus_vessel']}</div>
                    <div style="font-size: 12px; color: #6b7280;">{result['surplus_type']}</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
                <div>
                    <div style="font-size: 12px; color: #6b7280;">Credit Transfer</div>
                    <div style="font-size: 18px; font-weight: bold; color: #10b981;">{result['credit_transfer']:.3f}</div>
                </div>

                <div>
                    <div style="font-size: 12px; color: #6b7280;">Remaining Deficit</div>
                    <div style="font-size: 18px; font-weight: bold; color: #ef4444;">{result['remaining_deficit']:.3f}</div>
                </div>

                <div>
                    <div style="font-size: 12px; color: #6b7280;">Remaining Surplus</div>
                    <div style="font-size: 18px; font-weight: bold; color: #10b981;">{result['remaining_surplus']:.3f}</div>
                </div>

                <div>
                    <div style="font-size: 12px; color: #6b7280;">Estimated Cost</div>
                    <div style="font-size: 18px; font-weight: bold; color: #3b82f6;">${result['estimated_cost']:.2f}</div>
                </div>
            </div>

            <div style="margin-top: 15px; padding: 10px; background-color: {'#bbf7d0' if result['fully_covered'] else '#fde68a'}; border-radius: 5px;">
                <strong>{'✅ Deficit fully covered!' if result['fully_covered'] else '⚠️ Partial coverage achieved'}</strong>
                <div style="font-size: 14px; margin-top: 5px;">
                    {result['deficit_vessel']} can purchase {result['credit_transfer']:.3f} credits from {result['surplus_vessel']}
                    {'' if result['fully_covered'] else f', but still needs {result["remaining_deficit"]:.3f} more credits'}
                </div>
            </div>
        </div>
        """

        return result_html

    def display_dashboard(self):
        """Display the complete dashboard"""
        clear_output(wait=True)

        # Dashboard header
        header_html = f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); padding: 30px; border-radius: 10px; margin-bottom: 30px; color: white;">
            <h1 style="margin: 0; font-size: 32px;">🚢 Fleet Arbitrage Dashboard</h1>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">IMO GHG Compliance Trading Platform</p>
            <div style="margin-top: 15px; font-size: 14px; opacity: 0.8;">
                Last updated: {self.metrics['analysis_date']}
            </div>
        </div>
        """

        display(HTML(header_html))

        # Display stats cards
        display(HTML(self.create_stats_cards()))

        # Display charts side by side
        col1, col2 = widgets.Output(), widgets.Output()

        with col1:
            fig1 = self.create_compliance_chart()
            display(fig1)

        with col2:
            fig2 = self.create_intensity_chart()
            display(fig2)

        charts_container = widgets.HBox([col1, col2])
        display(charts_container)

        # Pooling simulator section
        pooling_html = f"""
        <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 30px 0;">
            <h2 style="margin-top: 0; color: #1f2937;">🔄 Pooling Simulator</h2>
            <p style="color: #6b7280; margin-bottom: 20px;">Select a deficit vessel and a surplus vessel to simulate credit pooling</p>
        </div>
        """

        display(HTML(pooling_html))

        # Pooling controls
        pooling_controls = widgets.HBox([
            self.deficit_select,
            self.surplus_select,
            self.simulate_btn,
            self.refresh_btn
        ])
        pooling_controls.layout.justify_content = 'space-between'
        display(pooling_controls)

        # Display pooling results
        if self.pooling_result:
            display(HTML(self.create_pooling_results()))

        # Liability map section
        liability_html = """
        <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 30px 0;">
            <h2 style="margin-top: 0; color: #1f2937;">🚢 Vessel Liability Map</h2>
            <p style="color: #6b7280; margin-bottom: 20px;">Filter vessels by compliance status and penalty risk level</p>
        </div>
        """

        display(HTML(liability_html))

        # Filter controls
        filter_controls = widgets.HBox([self.status_filter, self.risk_filter])
        filter_controls.layout.justify_content = 'flex-start'
        display(filter_controls)

        # Display liability table
        display(HTML(self.create_liability_table()))

        # Fleet statistics
        stats_html = f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0;">
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 12px; color: #6b7280;">Compliance Rate</div>
                <div style="font-size: 24px; font-weight: bold; color: #1f2937;">{self.metrics['compliance_rate']:.1f}%</div>
            </div>

            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 12px; color: #6b7280;">Avg GHG Intensity</div>
                <div style="font-size: 24px; font-weight: bold; color: #1f2937;">{self.metrics['avg_intensity']:.6f}</div>
            </div>

            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 12px; color: #6b7280;">Target Intensity</div>
                <div style="font-size: 24px; font-weight: bold; color: #1f2937;">{self.metrics['target_intensity']:.6f}</div>
            </div>

            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 12px; color: #6b7280;">Total Credits Available</div>
                <div style="font-size: 24px; font-weight: bold; color: #10b981;">{self.metrics['total_surplus']:.3f}</div>
            </div>
        </div>
        """

        display(HTML(stats_html))

        # Footer
        footer_html = """
        <div style="background: #1f2937; padding: 20px; border-radius: 10px; margin-top: 30px; color: white; text-align: center;">
            <p style="margin: 0; opacity: 0.8;">© 2024 MindX Compliance Engine • IMO GHG Regulation Dashboard</p>
            <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.6;">This dashboard implements the IMO 2026 5% GHG reduction target</p>
        </div>
        """

        display(HTML(footer_html))

# ============================================================================
# LAUNCH THE DASHBOARD
# ============================================================================

print("🚀 Launching Fleet Arbitrage Dashboard...")
print("="*60)

# Create and display dashboard
dashboard = FleetArbitrageDashboard(vessels_df, metrics)
dashboard.display_dashboard()

print("\n✅ Dashboard loaded successfully!")
print("\n📊 **Features Implemented:**")
print("   1. ✅ Liability Map - Categorizes ships by penalty risk level")
print("   2. ✅ Pooling Simulator - Interactive vessel pooling with real-time calculations")
print("   3. ✅ Compliance Distribution Charts")
print("   4. ✅ GHG Intensity Analysis")
print("   5. ✅ Real-time Fleet Statistics")
print("\n🔄 **How to use the Pooling Simulator:**")
print("   - Select a deficit vessel from the first dropdown")
print("   - Select a surplus vessel from the second dropdown")
print("   - Click 'Simulate Pooling' to see results")
print("   - Results show credit transfer, remaining deficit, and estimated cost")