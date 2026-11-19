"""The vendored trace-client-python 0.5.0 suite, run against interfaces/trace_client.py.

Drives the client against a real HTTP server on a loopback port -- the
standard library's own, so the tests have no more dependencies than the
client does."""
import decimal
import fractions
import json
import logging
import os
import pathlib
import re
import shutil
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interfaces.trace_client import (MAX_TAG_LENGTH, MAX_TAGS, TraceClient, __version__, allows_reporting,  # noqa: E402
                          environment_opts_out)  # noqa: E402
import interfaces.trace_client as trace_client_module  # noqa: E402

_ENV_VARS = ("TRACE_USAGE_REPORTING", "DO_NOT_TRACK")


class _Capture:
    def __init__(self):
        self.requests = []
        self.reply_status = 201
        self.location = None  # sent as a Location header when set
        self.arrived = threading.Event()
        self.release = threading.Event()
        self.release.set()  # by default answer at once


def _server(capture):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            capture.release.wait(10)
            capture.requests.append({
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "content_type": self.headers.get("Content-Type"),
                "user_agent": self.headers.get("User-Agent"),
                "body": body.decode("utf-8"),
            })
            self.send_response(capture.reply_status)
            if capture.location is not None:
                self.send_header("Location", capture.location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            capture.arrived.set()

        do_GET = do_POST  # a followed 301/302/303 arrives as a GET

        def log_message(self, *args):  # keep test output quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TraceClientTest(unittest.TestCase):
    def setUp(self):
        # The machine running the tests may itself have opted out; every
        # test starts from a clean environment and sets what it needs.
        scrubbed = {k: v for k, v in os.environ.items() if k not in _ENV_VARS}
        patcher = mock.patch.dict(os.environ, scrubbed, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.capture = _Capture()
        self.server = _server(self.capture)
        self.base_url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.log = []
        handler = logging.Handler()
        handler.emit = lambda record: self.log.append(record)
        self.handler = handler
        logging.getLogger("trace").addHandler(handler)
        logging.getLogger("trace").setLevel(logging.DEBUG)

    def tearDown(self):
        self.server.shutdown()
        logging.getLogger("trace").removeHandler(self.handler)

    def test_report_posts_the_event_to_the_metrics_endpoint_with_the_key(self):
        client = TraceClient(self.base_url + "/", "MyGame", "1.2.3", key="k-123")
        client.report("startup")
        self.assertTrue(self.capture.arrived.wait(5), "the report should reach the server")
        request = self.capture.requests[0]
        self.assertEqual("/api/metrics", request["path"], "a trailing slash on the base URL must not double up")
        self.assertEqual("Bearer k-123", request["authorization"])
        self.assertTrue(request["content_type"].startswith("application/json"))
        self.assertEqual({"application": "MyGame", "name": "startup", "tags": {"version": "1.2.3"}},
                         json.loads(request["body"]))
        client.close()

    def test_constructor_trims_the_base_url_application_and_key(self):
        client = TraceClient("  " + self.base_url + "/  ", "  MyGame  ", "1.2.3", key="  k-123  ")
        client.report("startup")
        self.assertTrue(self.capture.arrived.wait(5), "a padded base URL should still reach the server")
        request = self.capture.requests[0]
        self.assertEqual("/api/metrics", request["path"])
        self.assertEqual("Bearer k-123", request["authorization"], "the key is sent without its padding")
        self.assertEqual("MyGame", json.loads(request["body"])["application"])
        self.assertTrue(request["user_agent"].endswith("(MyGame)"), request["user_agent"])
        client.close()

    def test_a_base_url_with_a_path_keeps_it_in_front_of_the_endpoint(self):
        # A trace server behind a reverse proxy lives under a path prefix.
        client = TraceClient(self.base_url + "/trace//", "MyGame", "1.2.3", key="k")
        client.report("startup")
        self.assertTrue(self.capture.arrived.wait(5), "the report should reach the server")
        client.close()
        self.assertEqual("/trace/api/metrics", self.capture.requests[0]["path"],
                         "the prefix is kept and every trailing slash is dropped before /api/metrics")

    def test_non_ascii_names_and_tags_arrive_intact_as_utf8_json(self):
        client = TraceClient(self.base_url, "Spiel", "1.2.3", key="k")
        client.report("défi", tags={"ville": "Zürich", "emoji": "\U0001F3B2"})
        self.assertTrue(self.capture.arrived.wait(5), "the report should reach the server")
        client.close()
        request = self.capture.requests[0]
        self.assertEqual("application/json; charset=utf-8", request["content_type"])
        self.assertEqual({"application": "Spiel", "name": "défi",
                          "tags": {"ville": "Zürich", "emoji": "\U0001F3B2", "version": "1.2.3"}},
                         json.loads(request["body"]), "text the program reports is what the server reads")

    def test_an_application_name_a_header_cannot_carry_still_delivers_its_events(self):
        # The name goes in the User-Agent as well as the body, and a header
        # refuses most of Unicode and any line break.
        for application, shown in (("ゲーム", "???"), ("dice\U0001F3B2", "dice?"), ("two\nlines", "two?lines"),
                                   ("Zürich", "Zürich")):
            self.capture.requests.clear()
            client = TraceClient(self.base_url, application, "1.2.3", key="k")
            client.report("startup")
            client.close()
            self.assertEqual(1, len(self.capture.requests), "%r lost its event: %s"
                             % (application, [r.getMessage() for r in self.log]))
            request = self.capture.requests[0]
            self.assertEqual(application, json.loads(request["body"])["application"],
                             "the event keeps the exact name; only the header is approximate")
            self.assertEqual("trace-client-python/%s (%s)" % (__version__, shown), request["user_agent"])

    def test_report_carries_value_and_tags_when_given(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report("world-load", 2.5, {"seed": "42", "size": 'the "big" one'})
        self.assertTrue(self.capture.arrived.wait(5))
        self.assertEqual({"application": "MyGame", "name": "world-load", "value": 2.5,
                          "tags": {"seed": "42", "size": 'the "big" one', "version": "1.2.3"}},
                         json.loads(self.capture.requests[0]["body"]))
        client.close()

    def test_report_returns_before_the_server_answers(self):
        self.capture.release.clear()  # a server that never replies
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        before = time.monotonic()
        client.report("startup")
        elapsed = time.monotonic() - before
        self.assertLess(elapsed, 1.0, "report() took %.3fs; it must not wait on the network" % elapsed)
        self.capture.release.set()
        client.close()

    def test_report_does_not_raise_when_nothing_is_listening(self):
        probe = _server(_Capture())
        dead_port = probe.server_address[1]
        probe.shutdown()
        probe.server_close()
        client = TraceClient("http://127.0.0.1:%d" % dead_port, "MyGame", "1.2.3", key="k")
        client.report("startup")  # must not raise
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any("could not deliver" in r.getMessage() for r in self.log):
            time.sleep(0.05)
        client.close()
        self.assertTrue(any("could not deliver" in r.getMessage() for r in self.log), [r.getMessage() for r in self.log])
        self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_report_does_not_raise_when_the_server_rejects_the_key(self):
        self.capture.reply_status = 401
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="revoked")
        client.report("startup")
        self.assertTrue(self.capture.arrived.wait(5))
        client.close()
        self.assertTrue(any("answered 401" in r.getMessage() for r in self.log), [r.getMessage() for r in self.log])

    def test_a_201_is_delivered_without_logging_anything(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report("startup")
        self.assertTrue(self.capture.arrived.wait(5))
        client.close()
        self.assertEqual([], [r.getMessage() for r in self.log], "success is silent")

    def test_a_success_status_other_than_201_is_logged_as_not_delivered(self):
        for status in (200, 204):
            self.log.clear()
            self.capture.reply_status = status
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
            client.report("startup")
            client.close()  # joins the sender, so its log line is written by now
            self.assertTrue(any("answered %d" % status in r.getMessage() for r in self.log),
                            "only a 201 is success: %s" % [r.getMessage() for r in self.log])
            self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_a_redirect_is_logged_and_dropped_never_followed_with_the_key(self):
        target = _Capture()
        target_server = _server(target)
        self.addCleanup(target_server.shutdown)
        # Another port and another host name: what a redirect elsewhere looks like.
        self.capture.location = "http://localhost:%d/elsewhere" % target_server.server_address[1]
        for status in (301, 302, 303, 307, 308):
            self.log.clear()
            self.capture.reply_status = status
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="secret-key")
            client.report("startup")
            client.close()  # joins the sender, so its log line is written by now
            self.assertEqual([], target.requests,
                             "a %d must not carry the key to the host it names" % status)
            self.assertTrue(any("answered %d" % status in r.getMessage() for r in self.log),
                            "the log names the redirect that lost the event: %s"
                            % [r.getMessage() for r in self.log])
            self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_an_error_answer_whose_body_cannot_be_read_still_logs_its_status(self):
        class Unreadable(urllib.error.HTTPError):
            def read(self, *args):
                raise OSError("connection reset while reading the error body")

        answer = Unreadable(self.base_url + "/api/metrics", 503, "Service Unavailable", {}, None)
        crashes = []
        with mock.patch.object(threading, "excepthook", crashes.append), \
                mock.patch.object(trace_client_module._OPENER, "open", side_effect=answer):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
            client.report("startup")  # must not raise
            client.close()
        self.assertEqual([], crashes, "the sender thread must not die with a traceback on stderr")
        self.assertTrue(any("answered 503" in r.getMessage() for r in self.log), [r.getMessage() for r in self.log])
        self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_a_server_that_never_answers_costs_one_report_not_the_sender(self):
        # Each request is bounded by TIMEOUT_SECONDS (shortened here), so a
        # server that accepts the connection and then hangs drops that report
        # and the sender moves on to the next one.
        self.capture.release.clear()  # hold the first request unanswered
        with mock.patch.object(TraceClient, "TIMEOUT_SECONDS", 0.5):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
            client.report("first")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not any("could not deliver" in r.getMessage() for r in self.log):
                time.sleep(0.05)
            self.assertTrue(any("could not deliver" in r.getMessage() for r in self.log),
                            "a hung request must time out: %s" % [r.getMessage() for r in self.log])
            self.capture.release.set()
            client.report("second")
            client.close()
        self.assertIn("second", [json.loads(r["body"])["name"] for r in self.capture.requests],
                      "the sender is still alive after a request times out")
        self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_disabled_client_sends_nothing(self):
        for client in (TraceClient(self.base_url, "MyGame", "1.2.3", key="k", enabled=False),
                       TraceClient(self.base_url, "MyGame", "1.2.3"),
                       TraceClient(self.base_url, "MyGame", "1.2.3", key="  "),
                       TraceClient.disabled()):
            self.assertFalse(client.enabled)
            client.report("startup")
            client.close()
        self.assertFalse(self.capture.arrived.wait(0.3), "nothing should have been sent")
        self.assertEqual([], self.capture.requests)

    def test_disabled_reason_is_none_when_the_client_reports(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        self.assertTrue(client.enabled)
        self.assertIsNone(client.disabled_reason)
        client.close()
        self.assertFalse(client.enabled, "close() stops reporting")
        self.assertIsNone(client.disabled_reason, "but the reason describes how it was built")

    def test_disabled_reason_names_the_config_flag_or_the_missing_key(self):
        self.assertEqual("config", TraceClient(self.base_url, "MyGame", "1.2.3", key="k", enabled=False).disabled_reason)
        self.assertEqual("config", TraceClient.disabled().disabled_reason)
        self.assertEqual("no key", TraceClient(self.base_url, "MyGame", "1.2.3").disabled_reason)
        self.assertEqual("no key", TraceClient(self.base_url, "MyGame", "1.2.3", key="  ").disabled_reason)
        self.assertEqual("config", TraceClient(self.base_url, "MyGame", "1.2.3", enabled=False).disabled_reason,
                         "the config flag is checked before the key")

    def _assert_environment_disables(self, variable, value):
        with mock.patch.dict(os.environ, {variable: value}):
            self.assertTrue(environment_opts_out(), "%s=%r should opt out" % (variable, value))
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
            self.assertFalse(client.enabled, "%s=%r should disable the client" % (variable, value))
            self.assertEqual("environment", client.disabled_reason)
            client.report("startup")
            client.close()
        self.assertFalse(self.capture.arrived.wait(0.2), "%s=%r: nothing should have been sent" % (variable, value))
        self.assertEqual([], self.capture.requests)

    def test_TRACE_USAGE_REPORTING_off_disables_reporting_for_every_accepted_value(self):
        for value in ("off", "false", "0", "no", "OFF", "False", "No", " off "):
            self._assert_environment_disables("TRACE_USAGE_REPORTING", value)

    def test_TRACE_USAGE_REPORTING_set_to_anything_but_a_clear_on_disables(self):
        # Fails closed: a typo or an unfamiliar word is an opt-out, never consent.
        for value in ("disabled", "of", "nope", "2", "-1", "ture", "none", "never", '"true"'):
            self._assert_environment_disables("TRACE_USAGE_REPORTING", value)

    def test_DO_NOT_TRACK_disables_reporting_for_every_accepted_value(self):
        for value in ("1", "true", "yes", "TRUE", "Yes", " 1 "):
            self._assert_environment_disables("DO_NOT_TRACK", value)

    def test_DO_NOT_TRACK_set_to_anything_but_a_clear_no_disables(self):
        for value in ("on", "2", "y", "enabled", "please", "nein"):
            self._assert_environment_disables("DO_NOT_TRACK", value)

    def test_an_environment_that_cannot_be_read_disables_and_does_not_raise(self):
        class Unreadable(dict):
            def get(self, *args, **kwargs):
                raise PermissionError("no")
        self.assertTrue(environment_opts_out(Unreadable()))
        self.assertTrue(environment_opts_out({"DO_NOT_TRACK": 1}), "a value that is not a string opts out")
        with mock.patch.object(os, "environ", Unreadable()):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        self.assertFalse(client.enabled)
        self.assertEqual("environment", client.disabled_reason)

    def test_blank_or_clearly_on_environment_values_leave_the_program_setting_in_charge(self):
        for variable, value in (("TRACE_USAGE_REPORTING", "on"), ("TRACE_USAGE_REPORTING", ""),
                                ("TRACE_USAGE_REPORTING", "  "), ("TRACE_USAGE_REPORTING", "1"),
                                ("TRACE_USAGE_REPORTING", "TRUE"), ("TRACE_USAGE_REPORTING", " yes "),
                                ("DO_NOT_TRACK", "0"), ("DO_NOT_TRACK", ""), ("DO_NOT_TRACK", "false"),
                                ("DO_NOT_TRACK", "off"), ("DO_NOT_TRACK", " No ")):
            with mock.patch.dict(os.environ, {variable: value}):
                self.assertFalse(environment_opts_out(), "%s=%r is not an opt-out" % (variable, value))
                client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
                self.assertTrue(client.enabled, "%s=%r must not disable the client" % (variable, value))
                self.assertIsNone(client.disabled_reason)
                client.close()
        self.assertFalse(environment_opts_out(), "an unset variable is not an opt-out")

    def test_allows_reporting_is_true_only_for_True_or_a_clear_on_string(self):
        for setting in (True, "true", "on", "yes", "1", "TRUE", " On ", "Yes\n"):
            self.assertTrue(allows_reporting(setting), repr(setting))
        for setting in (False, None, "false", "off", "no", "0", "", "  ", "ture", '"false"', '"true"', "disabled",
                        1, 0, 1.0, [], ["true"], {"enabled": True}, object(), b"true"):
            self.assertFalse(allows_reporting(setting), repr(setting))

    def test_enabled_is_read_through_allows_reporting(self):
        for setting in ("false", "off", "no", '"false"', "ture", None, 1, "0"):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", enabled=setting)
            self.assertFalse(client.enabled, "enabled=%r must not report" % (setting,))
            self.assertEqual("config", client.disabled_reason, repr(setting))
        for setting in (True, "true", " ON ", "yes", "1"):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", enabled=setting)
            self.assertTrue(client.enabled, "enabled=%r reports" % (setting,))
            client.close()

    def test_environment_wins_over_the_config_flag_and_over_the_key(self):
        # enabled=True with a key, and the environment still says no.
        with mock.patch.dict(os.environ, {"TRACE_USAGE_REPORTING": "off"}):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", enabled=True)
            self.assertEqual("environment", client.disabled_reason)
            client.close()
        # enabled=False AND the environment: the environment is the reason.
        with mock.patch.dict(os.environ, {"DO_NOT_TRACK": "1"}):
            self.assertEqual("environment",
                             TraceClient(self.base_url, "MyGame", "1.2.3", key="k", enabled=False).disabled_reason)
            self.assertEqual("environment", TraceClient(self.base_url, "MyGame", "1.2.3").disabled_reason,
                             "the environment is checked before the key too")
        # Once the variable is gone, the program's own setting is back in charge.
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        self.assertTrue(client.enabled)
        client.report("startup")
        self.assertTrue(self.capture.arrived.wait(5))
        client.close()

    def test_environment_opts_out_accepts_an_explicit_mapping(self):
        self.assertTrue(environment_opts_out({"TRACE_USAGE_REPORTING": "off"}))
        self.assertTrue(environment_opts_out({"DO_NOT_TRACK": "yes"}))
        self.assertFalse(environment_opts_out({}))
        self.assertFalse(environment_opts_out({"DO_NOT_TRACK": "0", "TRACE_USAGE_REPORTING": "on"}))
        self.assertTrue(environment_opts_out({"TRACE_USAGE_REPORTING": "disabled"}))
        self.assertTrue(environment_opts_out({"DO_NOT_TRACK": "on"}))

    def test_environment_opts_out_reads_only_the_mapping_it_is_given(self):
        with mock.patch.dict(os.environ, {"DO_NOT_TRACK": "1"}):
            self.assertTrue(environment_opts_out(), "no mapping means the process environment")
            self.assertTrue(environment_opts_out(None), "None means the process environment")
            self.assertFalse(environment_opts_out({}), "an explicit mapping replaces the environment")
        self.assertTrue(environment_opts_out({"DO_NOT_TRACK": " TRUE "}),
                        "a mapping's values are trimmed and compared without case, like the environment's")
        self.assertTrue(environment_opts_out({"TRACE_USAGE_REPORTING": " No "}))

    def test_user_agent_names_the_client_version(self):
        from interfaces.trace_client import __version__
        self.assertEqual("0.5.0", __version__)
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report("startup")
        self.assertTrue(self.capture.arrived.wait(5))
        client.close()
        self.assertEqual("trace-client-python/0.5.0 (MyGame)", self.capture.requests[0]["user_agent"])



    def test_disabled_reason_values_are_the_exported_constants(self):
        from interfaces.trace_client import REASON_CONFIG, REASON_ENVIRONMENT, REASON_NO_KEY
        self.assertEqual(("environment", "config", "no key"), (REASON_ENVIRONMENT, REASON_CONFIG, REASON_NO_KEY),
                         "the strings the README documents, and the Java client sends the same")
        self.assertEqual(REASON_CONFIG, TraceClient.disabled().disabled_reason)
        self.assertEqual(REASON_NO_KEY, TraceClient(self.base_url, "MyGame", "1.2.3").disabled_reason)
        with mock.patch.dict(os.environ, {"DO_NOT_TRACK": "1"}):
            self.assertEqual(REASON_ENVIRONMENT, TraceClient(self.base_url, "MyGame", "1.2.3", key="k").disabled_reason)

    def test_report_does_not_raise_for_tags_that_are_not_a_mapping(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report("startup", tags=["kind"])  # must not raise
        client.report("command", tags=42)  # must not raise
        client.close()
        self.assertFalse(self.capture.arrived.wait(0.3), "a report that could not be built is dropped")
        self.assertEqual(["startup", "command"],
                         [r.args[0] for r in self.log if r.getMessage().startswith("[trace] could not queue")])
        self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_report_ignores_a_blank_name(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report("")
        client.report("   ")
        client.close()
        self.assertFalse(self.capture.arrived.wait(0.3))

    def test_report_does_not_raise_for_a_name_that_is_not_a_string(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report(123)  # must not raise
        client.close()
        self.assertFalse(self.capture.arrived.wait(0.3), "a report that could not be built is dropped")
        self.assertTrue(any("could not queue 123" in r.getMessage() for r in self.log),
                        [r.getMessage() for r in self.log])
        self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_a_base_url_without_a_scheme_is_logged_not_a_dead_thread(self):
        crashes = []
        with mock.patch.object(threading, "excepthook", crashes.append):
            client = TraceClient("trace.example.org", "MyGame", "1.2.3", key="k")
            client.report("startup")
            client.report("shutdown")
            client.close()
        self.assertEqual([], crashes, "the sender thread must not die with a traceback on stderr")
        failures = [r for r in self.log if "could not deliver" in r.getMessage()]
        self.assertEqual(2, len(failures), "each report is logged and the thread keeps going: %s"
                         % [r.getMessage() for r in self.log])
        self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_the_sender_thread_survives_an_unexpected_error_from_send(self):
        crashes = []
        real_send = TraceClient._send
        calls = []

        def send_that_fails_once(client, body):
            calls.append(body)
            if len(calls) == 1:
                raise RuntimeError("boom")
            real_send(client, body)

        with mock.patch.object(threading, "excepthook", crashes.append), \
                mock.patch.object(TraceClient, "_send", send_that_fails_once):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
            client.report("first")
            client.report("second")
            self.assertTrue(self.capture.arrived.wait(5), "a later report is still delivered")
            client.close()
        self.assertEqual([], crashes, "the sender thread must not die with a traceback on stderr")
        self.assertEqual(["second"], [json.loads(r["body"])["name"] for r in self.capture.requests])
        self.assertTrue(any("sender failed" in r.getMessage() and "boom" in r.getMessage() for r in self.log),
                        [r.getMessage() for r in self.log])
        self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log))

    def test_constructor_rejects_a_missing_base_url_or_application(self):
        for base_url, application in ((None, "MyGame"), (" ", "MyGame"), ("http://x", None), ("http://x", "")):
            with self.assertRaises(ValueError):
                TraceClient(base_url, application, "1.2.3")

    def test_constructor_rejects_a_missing_blank_or_overlong_version(self):
        for version in (None, "", "   ", "9" * (MAX_TAG_LENGTH + 1)):
            with self.assertRaises(ValueError, msg=repr(version)):
                TraceClient(self.base_url, "MyGame", version, key="k")
        with self.assertRaises(TypeError, msg="the version is required, not optional"):
            TraceClient(self.base_url, "MyGame", key="k")
        # Exactly the limit is fine, and so is an overlong-looking one that trims to it.
        TraceClient(self.base_url, "MyGame", "9" * MAX_TAG_LENGTH, enabled=False)
        TraceClient(self.base_url, "MyGame", "  " + "9" * MAX_TAG_LENGTH + "  ", enabled=False)

    def test_constructor_errors_name_the_argument_they_reject(self):
        overlong = "x" * (MAX_TAG_LENGTH + 1)
        cases = (
            ("base_url", lambda: TraceClient(" ", "MyGame", "1.2.3")),
            ("application", lambda: TraceClient(self.base_url, "", "1.2.3")),
            ("version", lambda: TraceClient(self.base_url, "MyGame", "  ")),
            ("version", lambda: TraceClient(self.base_url, "MyGame", overlong)),
            ("install_id", lambda: TraceClient(self.base_url, "MyGame", "1.2.3", install_id=overlong)),
        )
        for argument, construct in cases:
            with self.assertRaises(ValueError) as raised:
                construct()
            self.assertTrue(str(raised.exception).startswith(argument + " "),
                            "the only raising path should say which argument was wrong: %s" % raised.exception)
        with self.assertRaises(ValueError) as raised:
            TraceClient(self.base_url, "MyGame", overlong)
        self.assertIn(str(MAX_TAG_LENGTH), str(raised.exception), "an overlong value names the limit")

    def test_a_value_that_is_not_a_string_is_rejected_by_the_argument_it_was_given_for(self):
        cases = (
            ("base_url", "int", lambda **kw: TraceClient(5, "MyGame", "1.2.3", **kw)),
            ("application", "int", lambda **kw: TraceClient(self.base_url, 5, "1.2.3", **kw)),
            ("application", "bytes", lambda **kw: TraceClient(self.base_url, b"MyGame", "1.2.3", **kw)),
            ("version", "float", lambda **kw: TraceClient(self.base_url, "MyGame", 1.4, **kw)),
            ("key", "int", lambda **kw: TraceClient(self.base_url, "MyGame", "1.2.3", key=123, **kw)),
            ("install_id", "int", lambda **kw: TraceClient(self.base_url, "MyGame", "1.2.3", install_id=7, **kw)),
        )
        for argument, type_name, construct in cases:
            for options in ({}, {"enabled": False}):
                with self.assertRaises(ValueError, msg=argument) as raised:
                    construct(**options)
                self.assertEqual("%s must be a string, not %s" % (argument, type_name), str(raised.exception),
                                 "the only raising path should say which argument was wrong")

    def test_arguments_are_checked_before_the_environment_opt_out(self):
        with mock.patch.dict(os.environ, {"DO_NOT_TRACK": "1"}):
            with self.assertRaises(ValueError, msg="a bad argument should not hide behind the opt-out") as raised:
                TraceClient(self.base_url, "MyGame", 1.4, key="k")
            self.assertEqual("version must be a string, not float", str(raised.exception))
            with self.assertRaises(ValueError, msg="a missing argument should not hide behind the opt-out"):
                TraceClient(self.base_url, "", "1.2.3", key="k")

    def test_a_rejected_key_is_not_repeated_in_the_error(self):
        with self.assertRaises(ValueError) as raised:
            TraceClient(self.base_url, "MyGame", "1.2.3", key=123456789)
        self.assertNotIn("123456789", str(raised.exception), "an error can end up in a log; a key should not")

    def test_an_empty_value_of_another_type_keeps_its_old_meaning(self):
        # These never reached .strip(), so a program passing one already ran.
        for empty in (0, False, []):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key=empty, enabled=False)
            self.assertEqual("config", client.disabled_reason, repr(empty))
            self.assertEqual("no key", TraceClient(self.base_url, "MyGame", "1.2.3", key=empty).disabled_reason)
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id=empty)
            self.assertIsNone(client.install_id, repr(empty))
            client.close()
            with self.assertRaises(ValueError, msg=repr(empty)) as raised:
                TraceClient(self.base_url, "MyGame", empty)
            self.assertEqual("version is required", str(raised.exception))

    def test_the_sender_is_one_daemon_thread_named_after_the_application(self):
        before = set(threading.enumerate())
        client = TraceClient(self.base_url, "  MyGame  ", "1.2.3", key="k")
        try:
            started = [t for t in threading.enumerate() if t not in before]
            self.assertEqual(1, len(started), "one sender thread per client")
            sender = started[0]
            self.assertTrue(sender.daemon, "a daemon thread never keeps the host program from exiting")
            self.assertEqual("trace-client/MyGame", sender.name, "named after the trimmed application")
        finally:
            client.close()
        self.assertFalse(sender.is_alive(), "close() stops it")

    def test_a_disabled_client_starts_no_thread(self):
        before = set(threading.enumerate())
        clients = (TraceClient(self.base_url, "MyGame", "1.2.3", key="k", enabled=False),
                   TraceClient(self.base_url, "MyGame", "1.2.3"),
                   TraceClient.disabled())
        started = [t for t in threading.enumerate() if t not in before]
        self.assertEqual([], started, "a disabled client costs nothing")
        for client in clients:
            client.close()

    def test_report_tags_a_command_with_the_program_version_trimmed(self):
        client = TraceClient(self.base_url, "MyGame", " 2.0.0-SNAPSHOT ", key="k")
        client.report("command", tags={"name": "home"})
        self.assertTrue(self.capture.arrived.wait(5))
        client.close()
        self.assertEqual('{"application":"MyGame","name":"command",'
                         '"tags":{"name":"home","version":"2.0.0-SNAPSHOT"}}',
                         self.capture.requests[0]["body"])

    def test_report_an_events_own_version_tag_wins_over_the_program_version(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        tags = {"version": "9.9.9"}
        client.report("startup", tags=tags)
        self.assertTrue(self.capture.arrived.wait(5))
        client.close()
        self.assertEqual('{"application":"MyGame","name":"startup","tags":{"version":"9.9.9"}}',
                         self.capture.requests[0]["body"])
        self.assertEqual({"version": "9.9.9"}, tags, "the caller's dict is not modified")

    def test_report_never_modifies_the_callers_tags(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        tags = {"name": "home"}
        client.report("command", tags=tags)
        self.assertTrue(self.capture.arrived.wait(5))
        client.close()
        self.assertEqual({"name": "home"}, tags)
        self.assertEqual({"name": "home", "version": "1.2.3"},
                         json.loads(self.capture.requests[0]["body"])["tags"])

    def test_report_drops_none_tags_and_stringifies_the_rest_before_adding_the_version(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report("command", tags={"version": None, None: "orphan", "count": 7})
        self.assertTrue(self.capture.arrived.wait(5))
        self.assertEqual({"count": "7", "version": "1.2.3"}, json.loads(self.capture.requests[0]["body"])["tags"],
                         "a None version tag does not suppress the program's own version")
        client.close()

    def test_with_version_never_modifies_the_callers_mapping(self):
        from interfaces.trace_client import _with_version
        tags = {"name": "home"}
        merged = _with_version(tags, "1.2.3")
        self.assertEqual({"name": "home"}, tags)
        self.assertEqual({"name": "home", "version": "1.2.3"}, merged)
        self.assertEqual({"version": "1.2.3"}, _with_version(None, "1.2.3"))

    def test_json_drops_nan_and_none_tags(self):
        from interfaces.trace_client import _json
        body = json.loads(_json("App", "n", float("nan"), {"ok": "line\nbreak", "none": None, None: "x"}))
        self.assertEqual({"application": "App", "name": "n", "tags": {"ok": "line\nbreak"}}, body)

    def test_json_drops_infinite_values_but_keeps_zero(self):
        from interfaces.trace_client import _json
        for infinite in (float("inf"), float("-inf")):
            self.assertNotIn("value", json.loads(_json("App", "n", infinite, None)),
                             "JSON has no infinity; sending one would be rejected by the server")
        self.assertEqual(0, json.loads(_json("App", "n", 0, None))["value"], "zero is a value, not a missing one")

    def test_json_leaves_out_a_value_that_is_not_a_real_number_but_keeps_the_event(self):
        from interfaces.trace_client import _json
        for odd in ("3", "", True, False, b"3", [1], object(), 1j):
            body = json.loads(_json("App", "n", odd, None))
            self.assertEqual({"application": "App", "name": "n"}, body,
                             "%r is not a number the server can take; the event goes without it" % (odd,))

    def test_json_sends_other_real_numbers_as_floats(self):
        from interfaces.trace_client import _json
        self.assertEqual(1.5, json.loads(_json("App", "n", decimal.Decimal("1.5"), None))["value"])
        self.assertEqual(0.25, json.loads(_json("App", "n", fractions.Fraction(1, 4), None))["value"])
        self.assertEqual(7, json.loads(_json("App", "n", 7, None))["value"])
        for unsendable in (decimal.Decimal("NaN"), decimal.Decimal("sNaN"), decimal.Decimal("1e400"),
                           fractions.Fraction(10 ** 400)):
            self.assertNotIn("value", json.loads(_json("App", "n", unsendable, None)),
                             "%r has no finite float; it is left out rather than raising" % (unsendable,))

    def test_report_with_a_decimal_value_still_delivers_the_event(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        try:
            client.report("score", decimal.Decimal("2.5"))
            self.assertTrue(self.capture.arrived.wait(5), "a Decimal must not cost the event its place")
        finally:
            client.close()
        self.assertEqual(2.5, json.loads(self.capture.requests[0]["body"])["value"])

    def test_json_omits_tags_that_are_empty_after_cleaning_and_stringifies_the_rest(self):
        from interfaces.trace_client import _json
        self.assertNotIn("tags", json.loads(_json("App", "n", None, {"none": None})))
        self.assertNotIn("tags", json.loads(_json("App", "n", None, {})))
        self.assertEqual({"1": "2", "on": "True"}, json.loads(_json("App", "n", None, {1: 2, "on": True}))["tags"])

    def test_queue_is_bounded_and_drops_rather_than_grows(self):
        self.capture.release.clear()  # hold the sender on the first report
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        flood = TraceClient.QUEUE_CAPACITY * 3
        for _ in range(flood):
            client.report("flood")
        dropped = sum(1 for r in self.log if "queue full" in r.getMessage())
        self.assertGreaterEqual(dropped, flood - TraceClient.QUEUE_CAPACITY - 1,
                                "an unbounded queue would have accepted all %d" % flood)
        self.capture.release.set()
        client.close()

    def test_a_dropped_report_is_logged_at_debug_with_its_name(self):
        self.capture.release.clear()  # hold the sender on the first report
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report("filler")
        # Wait for the sender to take it: if it woke up between the last filler
        # and the overflow, the overflow would find a free slot and not be dropped.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not client._queue.empty():
            time.sleep(0.01)
        self.assertTrue(client._queue.empty(), "the sender should have picked up the first report")
        for _ in range(TraceClient.QUEUE_CAPACITY):
            client.report("filler")
        client.report("overflow-event")
        dropped = [r for r in self.log if r.getMessage() == "[trace] queue full, dropped overflow-event"]
        self.assertEqual(1, len(dropped),
                         "a drop names the event it dropped: %r" % [r.getMessage() for r in self.log])
        self.assertTrue(all(r.levelno == logging.DEBUG for r in self.log), "drops are logged at DEBUG only")
        self.capture.release.set()
        client.close()

    def test_reports_are_delivered_in_the_order_they_were_made(self):
        # One sender thread, one request at a time: the server sees events in
        # the order the program reported them.
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        for i in range(20):
            client.report("step", tags={"n": str(i)})
        client.close()
        self.assertEqual([str(i) for i in range(20)],
                         [json.loads(r["body"])["tags"]["n"] for r in self.capture.requests])

    def test_close_sends_what_was_just_queued_before_stopping(self):
        # A CLI reports once and exits at once. Without draining, the event
        # races the sender thread and is lost a good fraction of the time;
        # 30 back-to-back report()+close() pairs make that fraction visible.
        for i in range(30):
            client = TraceClient(self.base_url, "MyCli", "1.2.3", key="k")
            client.report("startup", tags={"run": str(i)})
            client.close()
        self.assertEqual(30, len(self.capture.requests), "every report()+close() pair must deliver")

    def test_close_still_returns_within_the_timeout_when_the_server_hangs(self):
        self.capture.release.clear()  # never answers
        client = TraceClient(self.base_url, "MyCli", "1.2.3", key="k")
        client.report("startup")
        before = time.monotonic()
        client.close(timeout=1.0)
        self.assertLess(time.monotonic() - before, 2.0, "draining must be bounded by the timeout")
        self.capture.release.set()

    def test_reports_still_queued_at_the_timeout_are_sent_after_close_returns(self):
        self.capture.release.clear()  # hold the sender on the first report
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        thread = client._thread
        client.report("seq", tags={"n": "0"})
        client.report("seq", tags={"n": "1"})
        client.close(timeout=0.5)
        self.assertTrue(thread.is_alive(), "close() returned on its timeout with the sender still busy")
        self.capture.release.set()
        thread.join(30)
        self.assertFalse(thread.is_alive(), "the sender stops once it reaches what close() queued behind them")
        self.assertEqual(["0", "1"], [json.loads(r["body"])["tags"]["n"] for r in self.capture.requests],
                         "reports queued before close() are not discarded when its timeout runs out")

    def test_close_is_prompt_and_idempotent(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.report("startup")
        before = time.monotonic()
        client.close()
        client.close()
        self.assertLess(time.monotonic() - before, TraceClient.TIMEOUT_SECONDS + 1)
        self.assertFalse(client.enabled)

    def test_report_after_close_sends_nothing(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        client.close()
        client.report("late")  # must not raise
        self.assertFalse(self.capture.arrived.wait(0.3), "a closed client should send nothing")
        self.assertEqual([], self.capture.requests)

    def test_close_on_a_full_queue_drops_the_oldest_report_and_still_stops_the_sender(self):
        self.capture.release.clear()  # hold the sender on the first report
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        thread = client._thread
        client.report("seq", tags={"n": "0"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not client._queue.empty():
            time.sleep(0.01)  # until the sender has taken it and is waiting on the server
        self.assertTrue(client._queue.empty(), "the sender should have picked up the first report")
        for n in range(1, TraceClient.QUEUE_CAPACITY + 2):  # one more than fits
            client.report("seq", tags={"n": str(n)})
        before = time.monotonic()
        client.close(timeout=0.5)
        self.assertLess(time.monotonic() - before, 2.0, "close() must stay bounded with a full queue")
        self.capture.release.set()
        thread.join(30)
        self.assertFalse(thread.is_alive(), "the sentinel must reach the sender even when the queue was full")
        sent = [json.loads(r["body"])["tags"]["n"] for r in self.capture.requests]
        self.assertEqual(TraceClient.QUEUE_CAPACITY, len(sent), "in flight + capacity - the displaced oldest")
        self.assertNotIn("1", sent, "the oldest queued report makes room for the sentinel")
        self.assertIn("0", sent)
        self.assertIn(str(TraceClient.QUEUE_CAPACITY), sent)

    def test_the_report_close_drops_from_a_full_queue_is_logged_at_debug(self):
        self.capture.release.clear()  # hold the sender on the first report
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        thread = client._thread
        client.report("in-flight")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not client._queue.empty():
            time.sleep(0.01)  # until the sender has taken it and is waiting on the server
        self.assertTrue(client._queue.empty(), "the sender should have picked up the first report")
        client.report("oldest-queued")
        for _ in range(TraceClient.QUEUE_CAPACITY - 1):
            client.report("filler")
        client.close(timeout=0.5)
        dropped = [r for r in self.log if r.getMessage().startswith("[trace] queue full at close, dropped ")]
        self.assertEqual(1, len(dropped),
                         "the report close() displaces is logged, like every other drop: %r"
                         % [r.getMessage() for r in self.log])
        self.assertIn("oldest-queued", dropped[0].getMessage(), "the log names the report it dropped")
        self.assertEqual(logging.DEBUG, dropped[0].levelno, "drops are logged at DEBUG only")
        self.capture.release.set()
        thread.join(30)

    # -- the per-installation ID ------------------------------------------

    def _tmpdir(self):
        directory = tempfile.mkdtemp(prefix="trace-install-")
        self.addCleanup(shutil.rmtree, directory, True)
        return directory

    def _sent_tags(self, client, name="startup", tags=None):
        client.report(name, tags=tags)
        self.assertTrue(self.capture.arrived.wait(5), "the report should reach the server")
        client.close()
        return json.loads(self.capture.requests[-1]["body"])["tags"]

    def test_no_install_id_and_no_file_sends_no_install_tag(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k")
        self.assertIsNone(client.install_id)
        self.assertEqual({"version": "1.2.3"}, self._sent_tags(client))

    def test_install_id_file_is_created_once_and_reused(self):
        path = os.path.join(self._tmpdir(), "nested", "deeper", "install-id")
        first = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id_file=path)
        self.assertEqual(str(uuid.UUID(first.install_id)), first.install_id, "a random UUID")
        with open(path, encoding="utf-8") as stored:
            self.assertEqual(first.install_id + "\n", stored.read(), "parent directories are created")
        first.close()
        second = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id_file=path)
        self.assertEqual(first.install_id, second.install_id, "the next run reuses it")
        self.assertEqual({"version": "1.2.3", "install": first.install_id}, self._sent_tags(second))

    def test_install_id_from_file_reads_the_first_valid_line(self):
        path = os.path.join(self._tmpdir(), "install-id")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n# not an id\n  my-own.id_1  \nsecond-id\n")
        self.assertEqual("my-own.id_1", TraceClient.install_id_from_file(path))
        with open(path, encoding="utf-8") as f:
            self.assertIn("# not an id", f.read(), "a file with an ID is never rewritten")

    def test_install_id_from_file_replaces_a_file_with_no_valid_line(self):
        path = os.path.join(self._tmpdir(), "install-id")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not an id\n" + "x" * (MAX_TAG_LENGTH + 1) + "\n")
        made = TraceClient.install_id_from_file(path)
        self.assertEqual(made, TraceClient.install_id_from_file(path))

    def test_install_id_from_file_writes_an_id_into_an_existing_empty_file(self):
        path = os.path.join(self._tmpdir(), "install-id")
        open(path, "w", encoding="utf-8").close()
        made = TraceClient.install_id_from_file(path)
        self.assertEqual(str(uuid.UUID(made)), made)
        with open(path, encoding="utf-8") as stored:
            self.assertEqual(made + "\n", stored.read(), "an empty file holds no ID, so one is written")

    def test_install_id_file_accepts_a_path_object(self):
        path = pathlib.Path(self._tmpdir()) / "nested" / "install-id"
        made = TraceClient.install_id_from_file(path)
        self.assertEqual(made + "\n", path.read_text(encoding="utf-8"), "os.PathLike is written like a str")
        self.assertEqual(made, TraceClient.install_id_from_file(path), "and read back like one")
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id_file=path)
        self.assertEqual(made, client.install_id, "the constructor takes the same os.PathLike")
        self.assertEqual({"version": "1.2.3", "install": made}, self._sent_tags(client))

    def test_unwritable_install_id_file_yields_an_in_memory_id_and_never_raises(self):
        # A path under a regular file cannot be created, even as root.
        blocker = os.path.join(self._tmpdir(), "a-file")
        with open(blocker, "w", encoding="utf-8") as f:
            f.write("x")
        path = os.path.join(blocker, "install-id")
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id_file=path)
        self.assertEqual(str(uuid.UUID(client.install_id)), client.install_id)
        self.assertFalse(os.path.exists(path))
        self.assertNotEqual(client.install_id, TraceClient.install_id_from_file(path),
                            "in memory: a new one each process")
        self.assertEqual({"version": "1.2.3", "install": client.install_id}, self._sent_tags(client))

    def test_unreadable_install_id_file_is_left_alone(self):
        directory = self._tmpdir()  # a directory cannot be read as a file
        made = TraceClient.install_id_from_file(directory)
        self.assertEqual(str(uuid.UUID(made)), made)
        self.assertTrue(os.path.isdir(directory))
        self.assertEqual([], os.listdir(directory))

    def test_install_id_from_file_never_raises_for_a_bad_path(self):
        for path in (None, "", "   ", 42):
            made = TraceClient.install_id_from_file(path)
            self.assertEqual(str(uuid.UUID(made)), made, repr(path))

    def test_install_id_from_file_takes_an_id_of_exactly_the_tag_length_limit(self):
        path = os.path.join(self._tmpdir(), "install-id")
        longest = "x" * MAX_TAG_LENGTH
        with open(path, "w", encoding="utf-8") as f:
            f.write("has/a/slash\nhas a space\n" + longest + "\n")
        self.assertEqual(longest, TraceClient.install_id_from_file(path),
                         "lines with other characters are skipped; %d characters is still an ID" % MAX_TAG_LENGTH)
        with open(path, encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("has/a/slash\n"), "a file with an ID is never rewritten")

    def test_install_id_file_failures_are_logged_at_debug_only(self):
        TraceClient.install_id_from_file(self._tmpdir())  # a directory cannot be read as a file
        missing = os.path.join(self._tmpdir(), "install-id")  # absent, so it is written -- and that fails
        with mock.patch("interfaces.trace_client.os.makedirs", side_effect=PermissionError("read-only")):
            TraceClient.install_id_from_file(missing)
        self.assertFalse(os.path.exists(missing))
        messages = [record.getMessage() for record in self.log]
        self.assertEqual(2, len(messages), messages)
        self.assertTrue(messages[0].startswith("[trace] could not read install ID file "), messages[0])
        self.assertTrue(messages[1].startswith("[trace] could not write install ID file "), messages[1])
        self.assertEqual({logging.DEBUG}, {record.levelno for record in self.log},
                         "an ID problem is never louder than DEBUG")

    def test_explicit_install_id_is_sent_as_given_without_the_files_character_rule(self):
        # The [A-Za-z0-9_.-] rule only picks an ID out of a file; an ID the
        # program passes itself is only trimmed and length-checked.
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="  mine/with spaces  ")
        self.assertEqual("mine/with spaces", client.install_id)
        self.assertEqual({"version": "1.2.3", "install": "mine/with spaces"}, self._sent_tags(client))

    def test_a_disabled_client_never_makes_up_or_writes_an_install_id(self):
        directory = self._tmpdir()
        path = os.path.join(directory, "install-id")
        with mock.patch.dict(os.environ, {"DO_NOT_TRACK": "1"}):
            by_environment = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id_file=path,
                                         install_id="explicit")
        by_config = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", enabled=False, install_id_file=path)
        by_no_key = TraceClient(self.base_url, "MyGame", "1.2.3", install_id_file=path)
        for client in (by_environment, by_config, by_no_key):
            self.assertIsNone(client.install_id, client.disabled_reason)
        self.assertEqual([], os.listdir(directory), "nothing is written")

    def test_explicit_install_id_is_trimmed_sent_and_wins_over_the_file(self):
        path = os.path.join(self._tmpdir(), "install-id")
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="  abc-123  ",
                             install_id_file=path)
        self.assertEqual("abc-123", client.install_id)
        self.assertFalse(os.path.exists(path), "the file is not consulted when an ID is given")
        self.assertEqual({"name": "home", "version": "1.2.3", "install": "abc-123"},
                         self._sent_tags(client, "command", {"name": "home"}))

    def test_blank_install_id_is_none_and_an_overlong_one_is_rejected(self):
        for blank in (None, "", "   "):
            client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id=blank)
            self.assertIsNone(client.install_id, repr(blank))
            client.close()
        with self.assertRaises(ValueError):
            TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="x" * (MAX_TAG_LENGTH + 1))
        with self.assertRaises(ValueError, msg="rejected even on a disabled client, like the version"):
            TraceClient(self.base_url, "MyGame", "1.2.3", enabled=False, install_id="x" * (MAX_TAG_LENGTH + 1))
        exact = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="x" * MAX_TAG_LENGTH)
        self.assertEqual("x" * MAX_TAG_LENGTH, exact.install_id)
        exact.close()

    def test_an_events_own_install_tag_wins(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="mine")
        tags = {"install": "theirs"}
        self.assertEqual({"install": "theirs", "version": "1.2.3"}, self._sent_tags(client, tags=tags))
        self.assertEqual({"install": "theirs"}, tags, "the caller's tags are never modified")

    def test_a_none_install_tag_does_not_suppress_the_installation_id(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="mine")
        self.assertEqual({"version": "1.2.3", "install": "mine"}, self._sent_tags(client, tags={"install": None}),
                         "a None tag is left out, so the client's own ID takes its place, as with version")
        self.capture.arrived.clear()
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="mine")
        self.assertEqual({"version": "1.2.3", "install": "7"}, self._sent_tags(client, tags={"install": 7}),
                         "any other value is the event's own, sent as a string, and wins")

    def test_install_tag_never_pushes_an_event_past_the_tag_cap(self):
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="mine")
        full = {"t%d" % i: "v" for i in range(MAX_TAGS - 1)}  # plus version = MAX_TAGS
        sent = self._sent_tags(client, tags=full)
        self.assertEqual(MAX_TAGS, len(sent))
        self.assertNotIn("install", sent)
        self.capture.arrived.clear()
        client = TraceClient(self.base_url, "MyGame", "1.2.3", key="k", install_id="mine")
        room = {"t%d" % i: "v" for i in range(MAX_TAGS - 2)}
        sent = self._sent_tags(client, tags=room)
        self.assertEqual(MAX_TAGS, len(sent))
        self.assertEqual("mine", sent["install"])


if __name__ == "__main__":
    unittest.main()
