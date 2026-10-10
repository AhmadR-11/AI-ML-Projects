"""Headless UI smoke testing suite for Streamlit application using AppTest."""

import sys
from streamlit.testing.v1 import AppTest


def assert_no_exception(at: AppTest, context: str) -> None:
    """Assert that AppTest did not encounter any unhandled exceptions."""
    if at.exception:
        msg = at.exception[0].message
        stack = "".join(at.exception[0].stack_trace) if hasattr(at.exception[0], "stack_trace") else ""
        print(f"\n[FAILURE] Exception during {context}: {msg}\n{stack}", file=sys.stderr)
        assert False, f"AppTest raised an exception during {context}: {msg}"


def main() -> None:
    print("Executing Phase 7 Streamlit AppTest smoke suite...")

    at = AppTest.from_file("app.py", default_timeout=180)

    # Scenario 1: Default run (Playground, Mud barrier 15x25)
    print("1. Running default view (Playground on Mud barrier)...")
    at.run()
    assert_no_exception(at, "Scenario 1: Default run")
    assert "grid_ascii" in at.session_state, "session_state['grid_ascii'] must exist"
    page_radio = at.sidebar.radio(key="page")
    assert page_radio.options == [
        "🧪 Playground",
        "⚖️ Compare",
        "📊 Benchmark",
        "💡 Insights",
    ], f"Unexpected page options: {page_radio.options}"

    # Scenario 2: Set preset to 'Trap (cup)'
    print("2. Changing preset to 'Trap (cup)'...")
    at.sidebar.selectbox(key="preset").select("Trap (cup)").run()
    assert_no_exception(at, "Scenario 2: Trap preset select")
    assert at.session_state["rows"] == 12, f"Expected rows=12, got {at.session_state.get('rows')}"
    assert at.session_state["cols"] == 20, f"Expected cols=20, got {at.session_state.get('cols')}"

    # Scenario 3: Switch page to '⚖️ Compare'
    print("3. Navigating to '⚖️ Compare'...")
    at.sidebar.radio(key="page").set_value("⚖️ Compare").run()
    assert_no_exception(at, "Scenario 3: Compare navigation")
    assert len(at.dataframe) >= 1, f"Expected at least 1 dataframe, got {len(at.dataframe)}"

    # Scenario 4: Turn on toggle key='cmp_race'
    print("4. Enabling 'cmp_race' toggle...")
    at.toggle(key="cmp_race").set_value(True).run()
    assert_no_exception(at, "Scenario 4: Race toggle")

    # Scenario 5: Turn on toggle key='cmp_precise'
    print("5. Enabling 'cmp_precise' toggle...")
    at.toggle(key="cmp_precise").set_value(True).run()
    assert_no_exception(at, "Scenario 5: Precise timing toggle")

    # Scenario 6: Set movement to '8-direction (diagonal)' and re-visit Compare and Playground
    print("6. Setting movement to '8-direction (diagonal)'...")
    at.sidebar.radio(key="movement").set_value("8-direction (diagonal)").run()
    assert_no_exception(at, "Scenario 6: Diagonal Compare run")
    at.sidebar.radio(key="page").set_value("🧪 Playground").run()
    assert_no_exception(at, "Scenario 6: Diagonal Playground run")

    # Scenario 7: Set preset 'Unsolvable'
    print("7. Selecting preset 'Unsolvable'...")
    at.sidebar.selectbox(key="preset").select("Unsolvable").run()
    assert_no_exception(at, "Scenario 7: Unsolvable Playground run")
    at.sidebar.radio(key="page").set_value("⚖️ Compare").run()
    assert_no_exception(at, "Scenario 7: Unsolvable Compare run")

    # Scenario 8: Set preset 'Random obstacles' with density 0.35 and seed 3
    print("8. Testing 'Random obstacles' with density 0.35...")
    at.sidebar.selectbox(key="preset").select("Random obstacles").run()
    assert_no_exception(at, "Scenario 8: Random obstacles select")
    at.sidebar.slider(key="density").set_value(0.35).run()
    assert_no_exception(at, "Scenario 8: Density slider adjustment")

    # Scenario 9: ASCII editor error handling and valid submission
    print("9. Testing ASCII editor error validation and updates...")
    at.sidebar.radio(key="page").set_value("🧪 Playground").run()
    assert_no_exception(at, "Scenario 9: Return to Playground")

    # Invalid input
    assert len(at.text_area) >= 1, "Expected ASCII text_area editor"
    at.text_area[0].input("S#\n#").run()
    at.button(key="apply_edit").click().run()
    assert_no_exception(at, "Scenario 9: Apply invalid edits")
    assert len(at.error) > 0, "Expected an error message to be displayed for invalid maze syntax"

    # Valid input (5 rows x 15 columns)
    valid_maze = (
        "S..............\n"
        "...............\n"
        "...............\n"
        "...............\n"
        "..............G"
    )
    at.text_area[0].input(valid_maze).run()
    at.button(key="apply_edit").click().run()
    assert_no_exception(at, "Scenario 9: Apply valid edits")
    assert at.session_state["grid_ascii"] == valid_maze, "grid_ascii was not updated with valid maze"

    # Scenario 10: Placeholder pages
    print("10. Testing placeholder pages '📊 Benchmark' and '💡 Insights'...")
    at.sidebar.radio(key="page").set_value("📊 Benchmark").run()
    assert_no_exception(at, "Scenario 10: Benchmark page")
    at.sidebar.radio(key="page").set_value("💡 Insights").run()
    assert_no_exception(at, "Scenario 10: Insights page")

    print("\nAll Phase 7 smoke tests passed!")


if __name__ == "__main__":
    main()
