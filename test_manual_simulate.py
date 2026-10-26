"""
Manual validation script for /api/simulate end-to-end integration.
"""
from demo.web_editor.app import app
from src.floor_plan import get_simple_room

def test_api_simulate_manual():
    client = app.test_client()
    grid = get_simple_room(20, 20, exit_width=2)

    response = client.post('/api/simulate', json={
        'grid': grid.tolist(),
        'num_agents': 80,
        'beta': 0.4,
        'D': 0.3,
        'k': 0.5,
        'max_steps': 60,
        'snapshot_every': 5,
        'seed': 42
    })

    assert response.status_code == 200, f"Failed: {response.get_json()}"
    data = response.get_json()

    print("=" * 75)
    print("  END-TO-END VALIDATION: /api/simulate PANIC FEEDBACK SIGNATURE")
    print("=" * 75)
    print(f"Room Geometry: 20x20 simple room (exit_width=2)")
    print(f"Parameters: N=80 agents, beta=0.4, D=0.3, k=0.5, max_steps=60, snap_every=5")
    print(f"Static Rational Baseline J*: {data['rational_J']:.4f} steps")
    print("-" * 75)
    print(f"{'Snapshot':<10} {'Timestep':<12} {'Worst-Case J(t)':<20} {'Mean Panic u':<16} {'Active Agents':<14}")
    print("-" * 75)

    for idx, f in enumerate(data['frames']):
        print(f"{idx+1:<10} t = {f['t']:<8} {f['worst_case']:>10.4f} steps    {f['mean_panic']:>12.4f}      {f['active_agents']:>6}")

    print("-" * 75)
    first_J = data['frames'][0]['worst_case']
    peak_J = max(f['worst_case'] for f in data['frames'])
    last_J = data['frames'][-1]['worst_case']
    print(f"Initial Snapshot J(t=5):   {first_J:.4f} steps")
    print(f"Peak Snapshot J(t):        {peak_J:.4f} steps")
    print(f"Final Snapshot J(t=60):    {last_J:.4f} steps")
    print(f"Static Rational Baseline J*: {data['rational_J']:.4f} steps")
    print(f"Max Feedback Increase:     +{((peak_J - data['rational_J']) / data['rational_J'] * 100):.2f}%")
    print(f"Final Evacuation Stats:    {data['final_stats']}")
    print("=" * 75)

if __name__ == "__main__":
    test_api_simulate_manual()
