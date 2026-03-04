"""Report CSS styles."""

REPORT_CSS = """
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary: #64748b;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont,
                'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }

        header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        header .subtitle {
            opacity: 0.9;
            font-size: 1rem;
        }

        nav {
            background: var(--card-bg);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        nav ul {
            list-style: none;
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            justify-content: center;
        }

        nav a {
            color: var(--text);
            text-decoration: none;
            font-weight: 500;
            padding: 0.25rem 0;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }

        nav a:hover {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }

        section {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        h2 {
            color: var(--primary);
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        }

        h3 {
            color: var(--text);
            margin: 1.5rem 0 1rem;
            font-size: 1.2rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg);
            font-weight: 600;
            color: var(--text);
        }

        tr:hover {
            background: var(--bg);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }

        .metric-card {
            background: var(--bg);
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }

        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
        }

        .metric-card .label {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }

        .plot-container {
            margin: 1.5rem auto;
            border: 1px solid var(--border);
            border-radius: 6px;
        }

        .plot-container h4 {
            background: var(--bg);
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
            border-bottom: 1px solid var(--border);
            text-align: center;
        }

        .plot-frame {
            width: 100%;
            border: none;
            display: block;
            margin: 0 auto;
            min-height: 400px;
            overflow: visible;
        }

        .significance {
            font-weight: 600;
        }

        .sig-high { color: var(--danger); }
        .sig-med { color: var(--warning); }
        .sig-low { color: var(--success); }

        .badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .badge-primary { background: #dbeafe; color: var(--primary); }
        .badge-success { background: #dcfce7; color: var(--success); }
        .badge-warning { background: #fef3c7; color: var(--warning); }
        .badge-danger { background: #fee2e2; color: var(--danger); }

        .summary-box {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 4px solid var(--primary);
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border-radius: 0 6px 6px 0;
        }

        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        .val-positive { color: green; }
        .val-warning { color: orange; }
        .val-match {
            background: #d4edda; color: #155724;
            padding: 2px 8px; border-radius: 4px; font-size: 0.85em;
        }
        .val-mismatch {
            background: #fff3cd; color: #856404;
            padding: 2px 8px; border-radius: 4px; font-size: 0.85em;
        }

        @media (max-width: 768px) {
            .container { padding: 1rem; }
            section { padding: 1rem; }
            nav ul { gap: 0.75rem; font-size: 0.9rem; }
        }
"""
