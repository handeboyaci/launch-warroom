"""Tests for the temporal leak detector."""

from warroom.validators.temporal_leak import scan_for_temporal_leaks


def test_clean_text():
  text = "Sponsor Amgen started trial NCT03600883 in 2018."
  res = scan_for_temporal_leaks(text)
  assert res.clean is True
  assert len(res.leaks) == 0


def test_known_event_blocklist():
  text = "Sotorasib was approved by the FDA."
  res = scan_for_temporal_leaks(text)
  assert res.clean is False
  assert any("approval" in leak for leak in res.leaks)

  text = "In the CodeBreaK 200 results..."
  res = scan_for_temporal_leaks(text)
  assert res.clean is False
  assert any("CodeBreaK 200" in leak for leak in res.leaks)


def test_explicit_dates_post_cutoff():
  # ISO format
  text = "The trial finished on 2021-05-15."
  res = scan_for_temporal_leaks(text)
  assert res.clean is False
  assert any("2021-05-15" in leak for leak in res.leaks)

  # Month YYYY format
  text = "Results were published in June 2021."
  res = scan_for_temporal_leaks(text)
  assert res.clean is False
  assert any("June" in leak for leak in res.leaks)


def test_explicit_dates_pre_cutoff():
  # ISO format exactly on cutoff (allowed is < cutoff)
  text = "The cutoff is 2021-01-14."
  res = scan_for_temporal_leaks(text)
  assert res.clean is True

  text = "Published in December 2020."
  res = scan_for_temporal_leaks(text)
  assert res.clean is True


def test_future_phrasing():
  text = "The drug went on to receive approval."
  res = scan_for_temporal_leaks(text)
  assert res.clean is False
  assert any("Future knowledge phrasing" in leak for leak in res.leaks)

  text = "Ultimately, the trial demonstrated..."
  res = scan_for_temporal_leaks(text)
  assert res.clean is False
  assert any("Retrospective knowledge phrasing" in leak for leak in res.leaks)


def test_standalone_year_post_cutoff():
  text = "Data later presented in 2023 was shocking."
  res = scan_for_temporal_leaks(text)
  assert res.clean is False
  assert any("2023" in leak for leak in res.leaks)


def test_january_2021_cutoff():
  text = "In January 2021..."
  res = scan_for_temporal_leaks(text)
  assert res.clean is False
  assert any("2021" in leak for leak in res.leaks)
