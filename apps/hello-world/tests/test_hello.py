from hello_world.main import main


def test_runs_offline(monkeypatch):
    monkeypatch.setenv('LOGFIRE_SEND_TO_LOGFIRE', 'false')
    main()
