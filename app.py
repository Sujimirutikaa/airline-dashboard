from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objs as go
import json
import numpy as np
import plotly.graph_objects as go
# Airport coordinates lookup (Lat, Lon)
airport_coordinates = {
    'JFK': (40.6413, -73.7781),
    'ATL': (33.6407, -84.4277),
    'LAX': (33.9416, -118.4085),
    'ORD': (41.9742, -87.9073),
    'DFW': (32.8998, -97.0403),
    'DEN': (39.8561, -104.6737),
    'SEA': (47.4502, -122.3088),
    'MIA': (25.7933, -80.2906),
    'BOS': (42.3656, -71.0096),
    'SFO': (37.6213, -122.3790),
    'LAS': (36.0840, -115.1537),
    'PHX': (33.4484, -112.0740),
    'IAH': (29.9902, -95.3368),
    'CLT': (35.2144, -80.9473),
    'MSP': (44.8821, -93.2218),
    'DTW': (42.2162, -83.3554),
    'PHL': (39.8744, -75.2424),
    'LGA': (40.7769, -73.8740),
    'BWI': (39.1776, -76.6684),
    'MDW': (41.7868, -87.7522)
}

def create_flight_route_map(filtered_df=None):
    
    if filtered_df is None:
        filtered_df = df
    
    # Calculate routes
    routes = filtered_df.groupby(['origin_airport', 'destination_airport']).agg({
        'is_delayed': ['count', 'sum'],
        'delay_minutes': 'mean'
    }).round(2)
    
    routes.columns = ['total_flights', 'delayed_flights', 'avg_delay']
    routes['delay_rate'] = (routes['delayed_flights'] / routes['total_flights'] * 100).round(1)
    routes = routes.reset_index()
    
    # DYNAMIC THRESHOLDS based on your data
    if len(routes) > 0:
        low_threshold = routes['delay_rate'].quantile(0.33)   # Bottom 33%
        high_threshold = routes['delay_rate'].quantile(0.67)  # Top 33%
    else:
        low_threshold, high_threshold = 5, 15
    
    fig = go.Figure()
    
    # Add flight routes with DYNAMIC thresholds
    for _, route in routes.iterrows():
        origin = route['origin_airport']
        dest = route['destination_airport']
        
        if origin in airport_coordinates and dest in airport_coordinates:
            origin_lat, origin_lon = airport_coordinates[origin]
            dest_lat, dest_lon = airport_coordinates[dest]
            
            # Dynamic color based on percentiles
            if route['delay_rate'] > high_threshold:
                line_color = 'red'
                line_width = 4
            elif route['delay_rate'] > low_threshold:
                line_color = 'orange'
                line_width = 3
            else:
                line_color = 'green'
                line_width = 2
            
            
            # Color based on delay rate
            if route['delay_rate'] > 15:
                line_color = 'red'
                line_width = 4
            elif route['delay_rate'] > 5:
                line_color = 'orange'
                line_width = 3
            else:
                line_color = 'green'
                line_width = 2
            
            # Add route line
            fig.add_trace(go.Scattergeo(
                lon=[origin_lon, dest_lon],
                lat=[origin_lat, dest_lat],
                mode='lines',
                line=dict(width=line_width, color=line_color),
                opacity=0.7,
                hovertemplate=f'<b>{origin} → {dest}</b><br>' +
                            f'Total Flights: {route["total_flights"]}<br>' +
                            f'Delayed Flights: {route["delayed_flights"]}<br>' +
                            f'Delay Rate: {route["delay_rate"]}%<br>' +
                            f'Avg Delay: {route["avg_delay"]:.1f} min<extra></extra>',
                showlegend=False
            ))
    
    # Add airport markers
    airports_in_data = list(set(filtered_df['origin_airport'].tolist() + filtered_df['destination_airport'].tolist()))
    
    for airport in airports_in_data:
        if airport in airport_coordinates:
            lat, lon = airport_coordinates[airport]
            
            # Calculate airport performance
            airport_flights = filtered_df[
                (filtered_df['origin_airport'] == airport) | 
                (filtered_df['destination_airport'] == airport)
            ]
            
            airport_delay_rate = (airport_flights['is_delayed'].sum() / len(airport_flights) * 100) if len(airport_flights) > 0 else 0
            
            # Color based on performance
            if airport_delay_rate > 15:
                marker_color = 'red'
                marker_size = 15
            elif airport_delay_rate > 5:
                marker_color = 'orange' 
                marker_size = 12
            else:
                marker_color = 'green'
                marker_size = 10
            
            fig.add_trace(go.Scattergeo(
                lon=[lon],
                lat=[lat],
                mode='markers+text',
                marker=dict(size=marker_size, color=marker_color, opacity=0.8),
                text=airport,
                textposition='top center',
                hovertemplate=f'<b>{airport} Airport</b><br>' +
                            f'Total Flights: {len(airport_flights)}<br>' +
                            f'Delay Rate: {airport_delay_rate:.1f}%<extra></extra>',
                showlegend=False
            ))
    
    # Update layout
    fig.update_geos(
        resolution=50,
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue',
        projection_type='natural earth'
    )
    
    fig.update_layout(
        title={
            'text': 'Flight Route Map - Delay Performance',
            'x': 0.5,
            'font': {'size': 20}
        },
        height=600,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

app = Flask(__name__)

# Load dataset
df = pd.read_csv('flight_data.csv')

def load_data():
    global df
    # Map your dataset columns to expected columns
    df['flight_date'] = pd.to_datetime(df['Departure_Time']).dt.date
    df['airline'] = df['Airline']
    df['origin_airport'] = df['Departure_Airport']
    df['destination_airport'] = df['Arrival_Airport']
    df['delay_minutes'] = pd.to_numeric(df['Delay_Minutes'], errors='coerce')
    df['is_delayed'] = df['delay_minutes'] > 0
    
    # Extract time components
    df['month'] = pd.to_datetime(df['Departure_Time']).dt.month
    df['hour'] = pd.to_datetime(df['Departure_Time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['Departure_Time']).dt.day_name()
    
    # Create delay reasons based on available data
    df['delay_reason'] = 'Unknown'
    df.loc[df['Weather_Impact'] == 1, 'delay_reason'] = 'Weather'
    df.loc[(df['is_delayed']) & (df['Weather_Impact'] == 0), 'delay_reason'] = 'Operational'
    df.loc[df['Flight_Status'] == 'Cancelled', 'delay_reason'] = 'Cancelled'

def filter_data(airline=None, airport=None, month=None, min_delay=None, max_delay=None):
    """Filter the dataset based on user inputs"""
    filtered_df = df.copy()
    
    if airline and airline != 'All':
        filtered_df = filtered_df[filtered_df['airline'] == airline]
    
    if airport and airport != 'All':
        filtered_df = filtered_df[
            (filtered_df['origin_airport'] == airport) | 
            (filtered_df['destination_airport'] == airport)
        ]
    
    if month and month != 'All':
        filtered_df = filtered_df[filtered_df['month'] == int(month)]
    
    if min_delay is not None:
        filtered_df = filtered_df[filtered_df['delay_minutes'] >= min_delay]
    
    if max_delay is not None:
        filtered_df = filtered_df[filtered_df['delay_minutes'] <= max_delay]
    
    return filtered_df

def get_delay_analysis(filtered_df=None):
    if filtered_df is None:
        filtered_df = df
        
    return {
        'total_flights': len(filtered_df),
        'delayed_flights': filtered_df['is_delayed'].sum(),
        'delay_rate': (filtered_df['is_delayed'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0,
        'avg_delay': filtered_df[filtered_df['is_delayed']]['delay_minutes'].mean() if filtered_df['is_delayed'].sum() > 0 else 0,
        'max_delay': filtered_df['delay_minutes'].max() if len(filtered_df) > 0 else 0
    }

def get_delay_by_airline(filtered_df=None):
    if filtered_df is None:
        filtered_df = df
        
    if len(filtered_df) == 0:
        return pd.DataFrame()
        
    airline_stats = filtered_df.groupby('airline').agg({
        'is_delayed': ['count', 'sum'],
        'delay_minutes': 'mean'
    }).round(2)
    airline_stats.columns = ['total_flights', 'delayed_flights', 'avg_delay']
    airline_stats['delay_rate'] = (airline_stats['delayed_flights'] / airline_stats['total_flights'] * 100).round(1)
    return airline_stats.reset_index()

def get_delay_by_airport(filtered_df=None):
    if filtered_df is None:
        filtered_df = df
        
    if len(filtered_df) == 0:
        return pd.DataFrame()
        
    origin_delays = filtered_df.groupby('origin_airport').agg({
        'is_delayed': ['count', 'sum'],
        'delay_minutes': 'mean'
    }).round(2)
    origin_delays.columns = ['total_flights', 'delayed_flights', 'avg_delay']
    origin_delays['delay_rate'] = (origin_delays['delayed_flights'] / origin_delays['total_flights'] * 100).round(1)
    return origin_delays.reset_index()

def get_delay_reasons(filtered_df=None):
    if filtered_df is None:
        filtered_df = df
    delay_reasons = filtered_df[filtered_df['is_delayed']]['delay_reason'].value_counts()
    return delay_reasons

def get_hourly_delay_pattern(filtered_df=None):
    if filtered_df is None:
        filtered_df = df
        
    if len(filtered_df) == 0:
        return pd.DataFrame()
        
    hourly_delays = filtered_df.groupby('hour').agg({
        'is_delayed': ['count', 'sum'],
        'delay_minutes': 'mean'
    }).round(2)
    hourly_delays.columns = ['total_flights', 'delayed_flights', 'avg_delay']
    hourly_delays['delay_rate'] = (hourly_delays['delayed_flights'] / hourly_delays['total_flights'] * 100).round(1)
    return hourly_delays

def get_monthly_trend(filtered_df=None):
    if filtered_df is None:
        filtered_df = df
        
    if len(filtered_df) == 0:
        return pd.DataFrame()
        
    monthly_delays = filtered_df.groupby('month').agg({
        'is_delayed': ['count', 'sum'],
        'delay_minutes': 'mean'
    }).round(2)
    monthly_delays.columns = ['total_flights', 'delayed_flights', 'avg_delay']
    monthly_delays['delay_rate'] = (monthly_delays['delayed_flights'] / monthly_delays['total_flights'] * 100).round(1)
    return monthly_delays
def generate_recommendations(filtered_df=None):
    if filtered_df is None:
        filtered_df = df
        
    recommendations = []
    
    # Check if we have any data
    if len(filtered_df) == 0:
        recommendations.append({
            'type': 'Data',
            'title': 'No Data Available',
            'description': 'No flights match the current filters. Try adjusting your filter criteria.',
            'priority': 'High'
        })
        return recommendations

    total_flights = len(filtered_df)
    delayed_flights = filtered_df['is_delayed'].sum()
    delay_rate = (delayed_flights / total_flights * 100) if total_flights > 0 else 0

    # Overall performance assessment
    if delay_rate == 0:
        recommendations.append({
            'type': 'Performance',
            'title': 'Excellent Performance!',
            'description': f'Perfect record! All {total_flights} flights in the filtered data were on time. Maintain current operational standards.',
            'priority': 'Low'
        })
    elif delay_rate < 10:
        recommendations.append({
            'type': 'Performance',
            'title': 'Good Performance',
            'description': f'Low delay rate of {delay_rate:.1f}% across {total_flights} flights. Minor optimizations may further improve performance.',
            'priority': 'Low'
        })
    elif delay_rate < 25:
        recommendations.append({
            'type': 'Performance',
            'title': 'Moderate Delays',
            'description': f'Delay rate of {delay_rate:.1f}% needs attention. {delayed_flights} out of {total_flights} flights were delayed.',
            'priority': 'Medium'
        })
    else:
        recommendations.append({
            'type': 'Performance',
            'title': 'High Delay Rate - Urgent Action Needed',
            'description': f'Critical delay rate of {delay_rate:.1f}%! {delayed_flights} out of {total_flights} flights delayed. Immediate operational review required.',
            'priority': 'High'
        })

    # Airline-specific recommendations
    try:
        airline_stats = get_delay_by_airline(filtered_df)
        if len(airline_stats) > 1:  # Multiple airlines
            worst_airline = airline_stats.loc[airline_stats['delay_rate'].idxmax()]
            best_airline = airline_stats.loc[airline_stats['delay_rate'].idxmin()]
            
            if worst_airline['delay_rate'] > best_airline['delay_rate'] + 5:
                recommendations.append({
                    'type': 'Airline Comparison',
                    'title': 'Performance Gap Between Airlines',
                    'description': f'{worst_airline["airline"]} has {worst_airline["delay_rate"]:.1f}% delay rate vs {best_airline["airline"]} at {best_airline["delay_rate"]:.1f}%. Focus improvement efforts on {worst_airline["airline"]}.',
                    'priority': 'Medium'
                })
        elif len(airline_stats) == 1:  # Single airline selected
            airline_name = airline_stats.iloc[0]['airline']
            airline_delay_rate = airline_stats.iloc[0]['delay_rate']
            recommendations.append({
                'type': 'Airline Focus',
                'title': f'{airline_name} Performance Analysis',
                'description': f'Filtered view shows {airline_name} with {airline_delay_rate:.1f}% delay rate across {airline_stats.iloc[0]["total_flights"]} flights.',
                'priority': 'Medium'
            })
    except Exception as e:
        print(f"Airline recommendation error: {e}")

    # Time-based recommendations
    try:
        hourly_pattern = get_hourly_delay_pattern(filtered_df)
        if len(hourly_pattern) > 0:
            peak_hour = hourly_pattern.loc[hourly_pattern['delay_rate'].idxmax()]
            if peak_hour['delay_rate'] > delay_rate + 10:  # Significantly higher than average
                recommendations.append({
                    'type': 'Schedule Optimization',
                    'title': f'Peak Delay Hour: {int(peak_hour.name)}:00',
                    'description': f'Hour {int(peak_hour.name)} shows {peak_hour["delay_rate"]:.1f}% delay rate, much higher than average. Consider redistributing {int(peak_hour["total_flights"])} flights or adding buffer time.',
                    'priority': 'High'
                })
    except Exception as e:
        print(f"Hourly recommendation error: {e}")

    # Weather-based recommendations
    try:
        weather_affected = filtered_df[filtered_df['Weather_Impact'] == 1]
        if len(weather_affected) > 0:
            weather_delay_rate = len(weather_affected) / len(filtered_df) * 100
            recommendations.append({
                'type': 'Weather Management',
                'title': f'Weather Impact: {weather_delay_rate:.1f}% of Flights',
                'description': f'{len(weather_affected)} flights affected by weather out of {total_flights} total. Consider enhanced weather monitoring and contingency planning.',
                'priority': 'Medium'
            })
    except Exception as e:
        print(f"Weather recommendation error: {e}")

    # Airport-specific recommendations
    try:
        airport_stats = get_delay_by_airport(filtered_df)
        if len(airport_stats) > 1:  # Multiple airports
            worst_airport = airport_stats.loc[airport_stats['delay_rate'].idxmax()]
            if worst_airport['delay_rate'] > delay_rate + 10:
                recommendations.append({
                    'type': 'Infrastructure',
                    'title': f'{worst_airport["origin_airport"]} Airport Congestion',
                    'description': f'{worst_airport["origin_airport"]} shows {worst_airport["delay_rate"]:.1f}% delay rate with {int(worst_airport["total_flights"])} flights. Review ground operations and capacity.',
                    'priority': 'High'
                })
        elif len(airport_stats) == 1:  # Single airport selected
            airport_name = airport_stats.iloc[0]['origin_airport']
            airport_delay_rate = airport_stats.iloc[0]['delay_rate']
            recommendations.append({
                'type': 'Airport Focus',
                'title': f'{airport_name} Airport Analysis',
                'description': f'Filtered analysis of {airport_name} shows {airport_delay_rate:.1f}% delay rate across {airport_stats.iloc[0]["total_flights"]} flights.',
                'priority': 'Medium'
            })
    except Exception as e:
        print(f"Airport recommendation error: {e}")

    return recommendations

'''def generate_recommendations(filtered_df=None):
    if filtered_df is None:
        filtered_df = df
        
    recommendations = []

    if filtered_df['is_delayed'].sum() == 0:
        recommendations.append({
            'type': 'Performance',
            'title': 'Excellent Performance!',
            'description': 'No significant delays detected in the filtered dataset. Continue current operational practices.',
            'priority': 'Low'
        })
        return recommendations

    try:
        hourly_pattern = get_hourly_delay_pattern(filtered_df)
        if len(hourly_pattern) > 0:
            peak_hours = hourly_pattern.nlargest(3, 'delay_rate').index.tolist()
            if peak_hours:
                recommendations.append({
                    'type': 'Schedule Optimization',
                    'title': 'Peak Delay Hours Identified',
                    'description': f'Hours {peak_hours} show highest delay rates in filtered data. Consider redistributing flights or adding buffer time.',
                    'priority': 'High'
                })
    except:
        pass

    try:
        airline_stats = get_delay_by_airline(filtered_df)
        if len(airline_stats) > 0:
            worst_airlines = airline_stats.nlargest(2, 'delay_rate')['airline'].tolist()
            if worst_airlines:
                recommendations.append({
                    'type': 'Airline Performance',
                    'title': 'Underperforming Airlines',
                    'description': f'{", ".join(worst_airlines)} have above-average delay rates in current selection.',
                    'priority': 'Medium'
                })
    except:
        pass

    return recommendations'''

@app.route('/')
def dashboard():
    load_data()
    
    # Get unique values for dropdowns
    airlines = ['All'] + sorted(df['airline'].unique().tolist())
    airports = ['All'] + sorted(df['origin_airport'].unique().tolist())
    months = ['All'] + [str(i) for i in sorted(df['month'].unique().tolist())]
    
    # Get filter parameters from URL
    selected_airline = request.args.get('airline', 'All')
    selected_airport = request.args.get('airport', 'All')
    selected_month = request.args.get('month', 'All')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    # Apply filters
    filtered_df = filter_data(selected_airline, selected_airport, selected_month, min_delay, max_delay)
    delay_stats = get_delay_analysis(filtered_df)
    
    return render_template('dashboard.html', 
                         stats=delay_stats,
                         airlines=airlines,
                         airports=airports,
                         months=months,
                         selected_airline=selected_airline,
                         selected_airport=selected_airport,
                         selected_month=selected_month,
                         min_delay=min_delay or 0,
                         max_delay=max_delay or int(df['delay_minutes'].max()))

@app.route('/api/airline_delays')
def api_airline_delays():
    airline = request.args.get('airline')
    airport = request.args.get('airport')
    month = request.args.get('month')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    filtered_df = filter_data(airline, airport, month, min_delay, max_delay)
    airline_stats = get_delay_by_airline(filtered_df)
    
    if len(airline_stats) == 0:
        return jsonify([])
    return jsonify(airline_stats.to_dict('records'))

@app.route('/api/airport_delays')
def api_airport_delays():
    airline = request.args.get('airline')
    airport = request.args.get('airport')
    month = request.args.get('month')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    filtered_df = filter_data(airline, airport, month, min_delay, max_delay)
    airport_stats = get_delay_by_airport(filtered_df)
    
    if len(airport_stats) == 0:
        return jsonify([])
    return jsonify(airport_stats.to_dict('records'))

@app.route('/api/delay_reasons')
def api_delay_reasons():
    airline = request.args.get('airline')
    airport = request.args.get('airport')
    month = request.args.get('month')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    filtered_df = filter_data(airline, airport, month, min_delay, max_delay)
    reasons = get_delay_reasons(filtered_df)
    return jsonify(reasons.to_dict())

@app.route('/api/hourly_pattern')
def api_hourly_pattern():
    airline = request.args.get('airline')
    airport = request.args.get('airport')
    month = request.args.get('month')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    filtered_df = filter_data(airline, airport, month, min_delay, max_delay)
    pattern = get_hourly_delay_pattern(filtered_df)
    
    if len(pattern) == 0:
        return jsonify({})
    return jsonify(pattern.to_dict('index'))

@app.route('/api/monthly_trend')
def api_monthly_trend():
    airline = request.args.get('airline')
    airport = request.args.get('airport')
    month = request.args.get('month')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    filtered_df = filter_data(airline, airport, month, min_delay, max_delay)
    trend = get_monthly_trend(filtered_df)
    
    if len(trend) == 0:
        return jsonify({})
    return jsonify(trend.to_dict('index'))

@app.route('/api/recommendations')
def api_recommendations():
    airline = request.args.get('airline')
    airport = request.args.get('airport')
    month = request.args.get('month')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    filtered_df = filter_data(airline, airport, month, min_delay, max_delay)
    recommendations = generate_recommendations(filtered_df)
    return jsonify(recommendations)

@app.route('/analytics')
def analytics():
    load_data()
    airlines = ['All'] + sorted(df['airline'].unique().tolist())
    airports = ['All'] + sorted(df['origin_airport'].unique().tolist())
    months = ['All'] + [str(i) for i in sorted(df['month'].unique().tolist())]
    return render_template('analytics.html', airlines=airlines, airports=airports, months=months)
@app.route('/flight_map')
def flight_map():
    load_data()
    
    # Get filter parameters
    selected_airline = request.args.get('airline', 'All')
    selected_airport = request.args.get('airport', 'All')
    selected_month = request.args.get('month', 'All')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    # Apply filters
    filtered_df = filter_data(selected_airline, selected_airport, selected_month, min_delay, max_delay)
    
    # Create map
    map_html = create_flight_route_map(filtered_df)
    
    # Get unique values for filters
    airlines = ['All'] + sorted(df['airline'].unique().tolist())
    airports = ['All'] + sorted(df['origin_airport'].unique().tolist())
    months = ['All'] + [str(i) for i in sorted(df['month'].unique().tolist())]
    
    return render_template('flight_map.html', 
                         map_html=map_html,
                         airlines=airlines,
                         airports=airports,
                         months=months,
                         selected_airline=selected_airline,
                         selected_airport=selected_airport,
                         selected_month=selected_month,
                         min_delay=min_delay or 0,
                         max_delay=max_delay or int(df['delay_minutes'].max()))
@app.route('/recommendations')
def recommendations():
    load_data()
    airline = request.args.get('airline')
    airport = request.args.get('airport')
    month = request.args.get('month')
    min_delay = request.args.get('min_delay', type=int)
    max_delay = request.args.get('max_delay', type=int)
    
    filtered_df = filter_data(airline, airport, month, min_delay, max_delay)
    recs = generate_recommendations(filtered_df)
    
    airlines = ['All'] + sorted(df['airline'].unique().tolist())
    airports = ['All'] + sorted(df['origin_airport'].unique().tolist())
    months = ['All'] + [str(i) for i in sorted(df['month'].unique().tolist())]
    
    return render_template('recommendations.html', recommendations=recs, airlines=airlines, airports=airports, months=months)

if __name__ == '__main__':
    load_data()
    app.run(debug=True)


