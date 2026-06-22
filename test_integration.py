"""V2 Integration tests."""
import json, os, subprocess, sys, tempfile


def test_cli_debug_search_smoke():
    """Run full CLI in debug mode and verify JSONL output."""
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    tmp.close()
    db_path = tmp.name.replace('.jsonl', '.db')
    try:
        cmd = [
            sys.executable, 'main.py',
            '--mode', 'debug', '--max-trials', '2', '--max-trial-frames', '3000',
            '--history', tmp.name, '--study-db', db_path, '--n-startup-trials', '1',
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0, completed.stdout + '\n' + completed.stderr
        with open(tmp.name, 'r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        trial_rows = [r for r in rows if r.get('record_type', 'trial') == 'trial']
        assert any(r.get('source') == 'baseline' for r in trial_rows)
        assert sum(1 for r in trial_rows if r.get('source') == 'tpe') == 2
    finally:
        os.unlink(tmp.name)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_compat_shim_imports_work():
    """The compat shim must still export all V1-accessible names."""
    from flappy_bird_dqn_auto_search import (
        run_trial, SearchDriver, BASELINE_CONFIG, HistoryManager,
        generate_summary, greedy_eval, compute_objective,
        FlappyBirdEnv, StateEncoder, DQNAgent,
    )
    assert run_trial is not None


def test_compat_shim_runs_baseline():
    """The compat CLI should work identically to main.py."""
    result = subprocess.run(
        [sys.executable, 'flappy_bird_dqn_auto_search.py', '--help'],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert '--baseline-only' in result.stdout


def test_main_help_includes_v3_cli_flags():
    result = subprocess.run(
        [sys.executable, 'main.py', '--help'],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert '--search-strategy' in result.stdout
    assert '--resume' in result.stdout
    assert '--recheck-topk' in result.stdout
    assert '--final-confirm' in result.stdout
    assert '--matrix' in result.stdout
