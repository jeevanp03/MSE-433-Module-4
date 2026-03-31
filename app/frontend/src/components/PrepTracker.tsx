import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Download,
  FileClock,
  Play,
  Plus,
  RotateCcw,
  Save,
  StopCircle,
  Trash2,
  Timer,
  Users,
  AlertTriangle,
} from 'lucide-react';
import {
  createSeedTrackerState,
  buildBlankPrepSession,
  PREP_NOTES_THRESHOLD_SEC,
  PREP_PHASE_CATALOG,
  PREP_REASON_CODES,
  PREP_ROLES,
  PREP_STAFF,
} from '../data/prepTrackerSeed';
import {
  type PrepEventLogEntry,
  type PrepPhase,
  type PrepPhaseSegment,
  type PrepPhaseStatus,
  type PrepReasonCode,
  type PrepRole,
  type PrepSession,
  type PrepSessionStatus,
  type PrepStaff,
} from '../types/prepTracker';

type TrackerState = {
  activeSessionId: string;
  sessions: PrepSession[];
};

type PhaseRuntime = {
  phase: PrepPhase;
  status: PrepPhaseStatus;
  totalDurationSec: number;
  firstStartAt: string | null;
  lastEndAt: string | null;
  isActive: boolean;
};

const STORAGE_KEY = 'mse433-prep-tracker-v1';

const toIsoNow = () => new Date().toISOString();

let eventCounter = 0;
const makeEventId = (prefix: string) => {
  eventCounter += 1;
  return `${prefix}-${eventCounter}-${Date.now().toString(36)}`;
};

let segmentCounter = 0;
const makeSegmentId = () => `seg-${segmentCounter}-${Date.now().toString(36)}`;

const pad2 = (n: number) => String(n).padStart(2, '0');

const formatClock = (seconds: number) => {
  const safe = Math.max(0, Math.floor(seconds));
  const hh = Math.floor(safe / 3600);
  const mm = Math.floor((safe % 3600) / 60);
  const ss = safe % 60;

  if (hh > 0) {
    return `${hh}:${pad2(mm)}:${pad2(ss)}`;
  }
  return `${pad2(mm)}:${pad2(ss)}`;
};

const formatDate = (value: string | null | undefined) => {
  if (!value) return 'Not started';
  const d = new Date(value);
  return d.toLocaleString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const formatTimeOnly = (value: string | null | undefined) => {
  if (!value) return '–';
  return new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const toDateTimeInput = (value: string | null | undefined) => {
  if (!value) return '';
  const date = new Date(value);
  const offsetMs = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - offsetMs).toISOString().slice(0, 16);
};

const fromDateTimeInput = (value: string) => {
  if (!value) return null;
  return new Date(value).toISOString();
};

const deriveSegmentDurationSec = (
  segment: PrepPhaseSegment,
  nowMs: number,
) => {
  if (!segment.endedAt) {
    return Math.max(0, Math.floor((nowMs - Date.parse(segment.startedAt)) / 1000));
  }
  return Math.max(0, Math.floor((Date.parse(segment.endedAt) - Date.parse(segment.startedAt)) / 1000));
};

const derivePhaseStatusFromSegments = (segments: PrepPhaseSegment[]): PrepPhaseStatus => {
  if (segments.length === 0) return 'not_started';
  return segments.some((s) => !s.endedAt) ? 'active' : 'completed';
};

const normalizeSession = (session: PrepSession, nowMs: number): PrepSession => {
  const nowPhaseStatus = session.phases.map((phase) => ({
    ...phase,
    status: derivePhaseStatusFromSegments(phase.segments),
  }));
  const status: PrepSessionStatus = session.accessStartedAt
    ? 'completed'
    : session.startedAt
      ? 'in_progress'
      : 'not_started';

  const totalPrepDurationSec = session.startedAt
    ? Math.max(0, Math.floor(((session.accessStartedAt ? Date.parse(session.accessStartedAt) : nowMs) - Date.parse(session.startedAt)) / 1000))
    : 0;

  return {
    ...session,
    status,
    phases: nowPhaseStatus,
    totalPrepDurationSec,
  };
};

const derivePhaseRuntime = (phase: PrepPhase, nowMs: number): PhaseRuntime => {
  const sorted = [...phase.segments].sort((a, b) => Date.parse(a.startedAt) - Date.parse(b.startedAt));
  const isActive = sorted.some((segment) => !segment.endedAt);
  const firstStartAt = sorted[0]?.startedAt ?? null;
  const lastEnded = [...sorted]
    .filter((s) => !!s.endedAt)
    .map((s) => Date.parse(s.endedAt as string))
    .sort((a, b) => a - b)
    .at(-1);

  const totalDurationSec = sorted.reduce((acc, segment) => acc + deriveSegmentDurationSec(segment, nowMs), 0);

  return {
    phase: { ...phase, status: derivePhaseStatusFromSegments(sorted) },
    status: derivePhaseStatusFromSegments(sorted),
    firstStartAt,
    lastEndAt: lastEnded != null ? new Date(lastEnded).toISOString() : null,
    isActive,
    totalDurationSec,
  };
};

const pickStartupSessionId = (sessions: PrepSession[]): string | null => {
  if (!sessions.length) return null;
  return (
    sessions.find((session) => session.status === 'not_started')?.id ??
    sessions.find((session) => session.status === 'in_progress')?.id ??
    sessions[0].id
  );
};

const pickNextActiveAfterDelete = (sessions: PrepSession[]): string | null =>
  pickStartupSessionId(sessions) ?? (sessions[0]?.id ?? null);

const safeLoadState = (): TrackerState => {
  const seedState = createSeedTrackerState();
  if (typeof localStorage === 'undefined') {
    return {
      activeSessionId: pickStartupSessionId(seedState.sessions) ?? seedState.activeSessionId,
      sessions: seedState.sessions.map((session) => normalizeSession(session, Date.now())),
    };
  }

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {
        activeSessionId: pickStartupSessionId(seedState.sessions) ?? seedState.activeSessionId,
        sessions: seedState.sessions.map((session) => normalizeSession(session, Date.now())),
      };
    }

    const parsed = JSON.parse(raw) as Partial<TrackerState>;
    if (!parsed || !Array.isArray(parsed.sessions) || parsed.sessions.length === 0) {
      throw new Error('invalid cache');
    }

    const sessions = parsed.sessions.map((session) => normalizeSession(session, Date.now()));
    const activeSessionId = parsed.activeSessionId && sessions.some((s) => s.id === parsed.activeSessionId)
      ? parsed.activeSessionId
      : pickStartupSessionId(sessions);

    if (!activeSessionId) {
      throw new Error('empty cache');
    }

    return {
      activeSessionId,
      sessions,
    };
  } catch {
    return {
      activeSessionId: pickStartupSessionId(seedState.sessions) ?? seedState.activeSessionId,
      sessions: seedState.sessions.map((session) => normalizeSession(session, Date.now())),
    };
  }
};

const eventActionLabel: Record<PrepEventLogEntry['action'], string> = {
  start: 'Start phase',
  stop: 'Stop phase',
  note_added: 'Note recorded',
  role_changed: 'Roles/staff updated',
  session_started: 'Session started',
  access_started: 'Access started',
};

function buildEventLogLine(entry: PrepEventLogEntry): string {
  return `${entry.action === 'start' ? '▸' : entry.action === 'stop' ? '◼' : '•'} ${
    eventActionLabel[entry.action]
  } ${entry.phaseId ? `(${entry.phaseId})` : ''} · ${formatTimeOnly(entry.timestamp)}`;
}

export default function PrepTracker() {
  const [trackerState, setTrackerState] = useState<TrackerState>(() => safeLoadState());
  const [tickMs, setTickMs] = useState(Date.now());
  const [noteDrafts, setNoteDrafts] = useState<Record<string, string>>({});
  const [demoEditMode, setDemoEditMode] = useState(false);

  const selectedSession = useMemo(
    () => trackerState.sessions.find((session) => session.id === trackerState.activeSessionId) ?? null,
    [trackerState],
  );

  useEffect(() => {
    const timer = window.setInterval(() => {
      setTickMs(Date.now());
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const hydrate = setTimeout(() => {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trackerState));
    }, 120);
    return () => window.clearTimeout(hydrate);
  }, [trackerState]);

  useEffect(() => {
    if (!selectedSession) return;
    const nextDrafts: Record<string, string> = {};
    for (const phase of selectedSession.phases) {
      nextDrafts[phase.id] = phase.freeTextNote ?? '';
    }
    setNoteDrafts(nextDrafts);
  }, [selectedSession?.id]);

  const orderedPhaseConfigs = PREP_PHASE_CATALOG;
  const reasonCodes: PrepReasonCode[] = PREP_REASON_CODES;
  const roles: PrepRole[] = PREP_ROLES;
  const staff: PrepStaff[] = PREP_STAFF;

  const activeSessionSummary = useMemo(() => {
    if (!selectedSession) return null;
    const orderedPhaseRuntimes = orderedPhaseConfigs
      .map((phaseConfig) => {
        const phase = selectedSession.phases.find((p) => p.key === phaseConfig.key);
        if (!phase) return null;
        return { phaseConfig, runtime: derivePhaseRuntime(phase, tickMs) };
      })
      .filter((item): item is { phaseConfig: typeof orderedPhaseConfigs[number]; runtime: PhaseRuntime } => item !== null);

    const activePhases = orderedPhaseRuntimes.filter((item) => item.runtime.isActive);
    const phaseNotes = selectedSession.phases.filter(
      (p) => p.freeTextNote.trim().length > 0 || p.reasonCodes.length > 0,
    );
    const otherActive = trackerState.sessions.some(
      (session) => session.id !== selectedSession.id && session.status === 'in_progress',
    );

    return {
      orderedPhaseRuntimes,
      activePhases,
      phaseNotes,
      otherActive,
    };
  }, [selectedSession, tickMs, trackerState.sessions]);

  const hasNoSession = !selectedSession;
  const overallLabel = selectedSession?.status === 'not_started'
    ? 'Not started'
    : selectedSession?.status === 'in_progress'
      ? 'In progress'
      : 'Completed';

  const canStartSession = !hasNoSession && selectedSession.status === 'not_started';
  const canAccess = !hasNoSession && selectedSession?.status === 'in_progress' && !!selectedSession.startedAt && !selectedSession.accessStartedAt;
  const canEditPhases = !hasNoSession && ((selectedSession?.status === 'in_progress' || demoEditMode));
  const canSave = !hasNoSession && selectedSession?.status === 'in_progress' && !!selectedSession.startedAt;

  const closeOpenSegments = (session: PrepSession, now: string, eventDetail?: string): PrepSession => {
    let withEvents = session;
    const closedPhases = session.phases.map((phase) => {
      const hasOpen = phase.segments.some((segment) => !segment.endedAt);
      if (!hasOpen) return phase;

      withEvents = appendEvent(withEvents, 'stop', phase.id, eventDetail);
      return {
        ...phase,
        status: 'completed',
        segments: phase.segments.map((segment) => (segment.endedAt ? segment : { ...segment, endedAt: now })),
      };
    });

    return {
      ...withEvents,
      phases: closedPhases,
    };
  };

  const updateActiveSession = (updater: (session: PrepSession) => PrepSession) => {
    setTrackerState((prev) => {
      const sessions = prev.sessions.map((session) =>
        session.id === prev.activeSessionId ? normalizeSession(updater(session), Date.now()) : session,
      );
      const activeSessionId = sessions.some((s) => s.id === prev.activeSessionId)
        ? prev.activeSessionId
        : sessions[0]?.id ?? prev.activeSessionId;
      return { ...prev, sessions, activeSessionId };
    });
  };

  const setActiveSession = (sessionId: string) => {
    setTrackerState((prev) => ({ ...prev, activeSessionId: sessionId }));
  };

  const appendEvent = (session: PrepSession, action: PrepEventLogEntry['action'], phaseId?: string, detail?: string) => ({
    ...session,
    eventLog: [...session.eventLog, {
      id: makeEventId('evt'),
      timestamp: toIsoNow(),
      action,
      phaseId,
      recorderId: session.recorderId,
      detail,
    }],
  });

  const startSession = () => {
    if (!selectedSession || !canStartSession) return;
    const now = toIsoNow();
    updateActiveSession((session) => {
      if (session.startedAt) return session;
      return {
        ...appendEvent(session, 'session_started'),
        startedAt: now,
      };
    });
  };

  const markAccessStarted = () => {
    if (!selectedSession) return;
    const now = toIsoNow();
    updateActiveSession((session) => {
      let next = session.startedAt ? session : appendEvent({ ...session, startedAt: now }, 'session_started');
      next = closeOpenSegments(next, now, 'Auto-stopped at access mark');

      next = {
        ...next,
        accessStartedAt: next.accessStartedAt ?? now,
      };
      if (next.status !== 'completed') {
        next = {
          ...next,
          status: 'completed',
        };
      }

      return appendEvent(next, 'access_started');
    });
  };

  const saveSession = () => {
    if (!selectedSession || !canSave) return;
    const now = toIsoNow();
    updateActiveSession((session) => {
      if (session.accessStartedAt) {
        return {
          ...session,
          status: 'completed',
        };
      }

      const startedSession = session.startedAt ? session : appendEvent({ ...session, startedAt: now }, 'session_started');
      const withClosedPhases = closeOpenSegments(startedSession, now, 'Auto-stopped at manual save');
      const completed = {
        ...withClosedPhases,
        accessStartedAt: now,
        status: 'completed',
      };
      return appendEvent(completed, 'access_started', undefined, 'Manual complete action');
    });
  };

  const resetToDemo = () => {
    const seedState = createSeedTrackerState(Date.now());
    localStorage.removeItem(STORAGE_KEY);
    setTrackerState({
      activeSessionId: seedState.activeSessionId,
      sessions: seedState.sessions.map((session) => normalizeSession(session, Date.now())),
    });
  };

  const createNewProcedureSession = () => {
    const next = buildBlankPrepSession(Date.now());
    setTrackerState((prev) => ({
      ...prev,
      activeSessionId: next.id,
      sessions: [next, ...prev.sessions].sort((a, b) => Date.parse(b.dateTime) - Date.parse(a.dateTime)),
    }));
  };

  const deleteProcedureSession = (sessionId: string) => {
    if (!selectedSession) return;
    if (!window.confirm('Delete this procedure record? This cannot be undone.')) return;

    setTrackerState((prev) => {
      const sessions = prev.sessions.filter((session) => session.id !== sessionId);
      if (sessions.length === 0) {
        return prev;
      }
      const nextActiveSessionId = sessionId === prev.activeSessionId
        ? pickNextActiveAfterDelete(sessions)
        : prev.activeSessionId;

      return {
        ...prev,
        sessions,
        activeSessionId: nextActiveSessionId ?? prev.activeSessionId,
      };
    });
  };

  const startPhase = (phase: PrepPhase) => {
    if (!selectedSession || !canEditPhases || selectedSession.status === 'completed') return;
    if (phase.segments.some((segment) => !segment.endedAt)) return;
    const now = toIsoNow();
    updateActiveSession((session) => {
      const updatedPhases = session.phases.map((candidate) => {
        if (candidate.id !== phase.id) return candidate;
        return {
          ...candidate,
          status: 'active',
          segments: [
            ...candidate.segments,
            { id: makeSegmentId(), startedAt: now },
          ],
        };
      });
      return appendEvent({ ...session, phases: updatedPhases }, 'start', phase.id);
    });
  };

  const stopPhase = (phase: PrepPhase) => {
    if (!selectedSession || !canEditPhases || selectedSession.status === 'completed') return;
    if (!phase.segments.some((segment) => !segment.endedAt)) return;
    const now = toIsoNow();
    updateActiveSession((session) => {
      const updatedPhases = session.phases.map((candidate) => {
        if (candidate.id !== phase.id) return candidate;
        const stoppedSegments = candidate.segments.map((segment) => (segment.endedAt ? segment : { ...segment, endedAt: now }));
        return {
          ...candidate,
          status: 'completed',
          segments: stoppedSegments,
        };
      });
      return appendEvent({ ...session, phases: updatedPhases }, 'stop', phase.id);
    });
  };

  const updatePhasePrimaryRole = (phase: PrepPhase, roleId: string) => {
    if (!selectedSession || !canEditPhases || selectedSession.status === 'completed') return;
    updateActiveSession((session) => ({
      ...appendEvent(
        {
          ...session,
          phases: session.phases.map((candidate) => {
            if (candidate.id !== phase.id) return candidate;
            const nextSupportingRoles = candidate.supportingRoleIds.filter((role) => role !== roleId);
            return {
              ...candidate,
              primaryRoleId: roleId,
              supportingRoleIds: nextSupportingRoles,
            };
          }),
        },
        'role_changed',
        phase.id,
      ),
      status: session.status,
    }));
  };

  const toggleSupportingRole = (phase: PrepPhase, roleId: string) => {
    if (!selectedSession || !canEditPhases || selectedSession.status === 'completed') return;
    updateActiveSession((session) => ({
      ...appendEvent(
        {
          ...session,
          phases: session.phases.map((candidate) => {
            if (candidate.id !== phase.id) return candidate;
            const isActive = candidate.supportingRoleIds.includes(roleId);
            const supportingRoleIds = isActive
              ? candidate.supportingRoleIds.filter((id) => id !== roleId)
              : [...candidate.supportingRoleIds, roleId];
            return { ...candidate, supportingRoleIds };
          }),
        },
        'role_changed',
        phase.id,
      ),
      status: session.status,
    }));
  };

  const toggleResponsibleStaff = (phase: PrepPhase, staffId: string) => {
    if (!selectedSession || !canEditPhases || selectedSession.status === 'completed') return;
    updateActiveSession((session) => ({
      ...appendEvent(
        {
          ...session,
          phases: session.phases.map((candidate) => {
            if (candidate.id !== phase.id) return candidate;
            const active = candidate.responsibleStaffIds.includes(staffId);
            const responsibleStaffIds = active
              ? candidate.responsibleStaffIds.filter((id) => id !== staffId)
              : [...candidate.responsibleStaffIds, staffId];
            return { ...candidate, responsibleStaffIds };
          }),
        },
        'role_changed',
        phase.id,
      ),
      status: session.status,
    }));
  };

  const toggleReasonCode = (phase: PrepPhase, reasonId: string) => {
    if (!selectedSession || !canEditPhases) return;
    updateActiveSession((session) => ({
      ...appendEvent(
        {
          ...session,
          phases: session.phases.map((candidate) => {
            if (candidate.id !== phase.id) return candidate;
            const active = candidate.reasonCodes.includes(reasonId);
            const reasonCodes = active
              ? candidate.reasonCodes.filter((id) => id !== reasonId)
              : [...candidate.reasonCodes, reasonId];
            const extraordinaryFlag = reasonCodes.length > 0 ? true : candidate.extraordinaryFlag && candidate.freeTextNote.length > 0;
            return { ...candidate, reasonCodes, extraordinaryFlag };
          }),
        },
        'note_added',
        phase.id,
      ),
      status: session.status,
    }));
  };

  const toggleExtraordinary = (phase: PrepPhase) => {
    if (!selectedSession || !canEditPhases || selectedSession.status === 'completed') return;
    updateActiveSession((session) => ({
      ...appendEvent(
        {
          ...session,
          phases: session.phases.map((candidate) => {
            if (candidate.id !== phase.id) return candidate;
            return {
              ...candidate,
              extraordinaryFlag: !candidate.extraordinaryFlag,
            };
          }),
        },
        'note_added',
        phase.id,
      ),
      status: session.status,
    }));
  };

  const commitNote = (phase: PrepPhase) => {
    if (!selectedSession || selectedSession.status === 'completed') return;
    const draft = noteDrafts[phase.id] ?? '';
    if (draft === phase.freeTextNote) return;

    updateActiveSession((session) => ({
      ...appendEvent(
        {
          ...session,
          phases: session.phases.map((candidate) => {
            if (candidate.id !== phase.id) return candidate;
            return { ...candidate, freeTextNote: draft };
          }),
        },
        'note_added',
        phase.id,
        draft.slice(0, 80),
      ),
      status: session.status,
    }));
  };

  const exportCurrentSession = () => {
    if (!selectedSession) return;
    const payload = {
      exportedAt: toIsoNow(),
      session: selectedSession,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedSession.caseId}_prep_session.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const updateSessionStart = (value: string) => {
    if (!selectedSession) return;
    const parsed = fromDateTimeInput(value);
    if (!parsed) return;
    updateActiveSession((session) => ({
      ...session,
      startedAt: parsed,
      status: session.accessStartedAt ? 'completed' : session.startedAt ? 'in_progress' : 'not_started',
    }));
  };

  const updateSessionAccess = (value: string) => {
    if (!selectedSession) return;
    const parsed = fromDateTimeInput(value);
    if (!parsed) return;
    updateActiveSession((session) => ({
      ...session,
      accessStartedAt: parsed,
      status: parsed ? 'completed' : session.status,
    }));
  };

  const sessionRows = useMemo(
    () =>
      [...trackerState.sessions].sort((a, b) => Date.parse(b.dateTime) - Date.parse(a.dateTime)),
    [trackerState.sessions],
  );

  const selectedRuntime = useMemo(() => (selectedSession ? {
    session: normalizeSession(selectedSession, tickMs),
    summary: selectedSession.phases.reduce<Record<string, PhaseRuntime>>((acc, phase) => {
      acc[phase.id] = derivePhaseRuntime(phase, tickMs);
      return acc;
    }, {}),
  } : null), [selectedSession, tickMs]);

  if (hasNoSession || !selectedRuntime) {
    return (
      <div className="text-sm text-gray-600">
        No prep tracker case loaded. Use reset to load demo cases.
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#1e40af]/10 rounded-lg">
            <FileClock size={24} className="text-[#1e40af]" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Prep/Intubation Tracker</h1>
            <p className="text-sm text-gray-500">Clinically safe, tap-efficient phase tracking with overlap support</p>
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs text-gray-600 bg-white border border-gray-200 rounded-lg px-3 py-2 cursor-pointer">
          <input
            type="checkbox"
            checked={demoEditMode}
            onChange={(event) => setDemoEditMode(event.target.checked)}
          />
          Demo edit mode
        </label>
      </div>

      <div className="flex items-center gap-2 mb-4 overflow-x-auto">
        {sessionRows.map((session) => (
          <div key={session.id} className="relative">
            <button
              onClick={() => setActiveSession(session.id)}
              className={`w-full px-3 py-2 rounded-lg text-sm border min-h-10 text-left ${
                session.id === selectedSession.id
                  ? 'bg-[#1e40af] text-white border-[#1e40af]'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="font-semibold">{session.caseId}</div>
              <div className="text-[11px] opacity-80">{formatDate(session.dateTime)}</div>
              <div className="text-[11px] uppercase tracking-wide mt-1 font-bold">
                {session.status === 'not_started' ? 'Not started' : session.status === 'in_progress' ? 'In progress' : 'Completed'}
              </div>
            </button>
            <button
              onClick={(event) => {
                event.stopPropagation();
                deleteProcedureSession(session.id);
              }}
              className="absolute top-2 right-2 p-1 rounded-md border border-rose-200 bg-white text-rose-600 hover:bg-rose-50"
              aria-label={`Delete ${session.caseId}`}
              title="Delete procedure"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 space-y-6">
          <motion.div
            className="bg-white rounded-xl border border-gray-200 p-5"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Case session shell</h2>
                <p className="text-sm text-gray-500">Case {selectedSession.caseId}</p>
              </div>
              <span
                className={`text-xs font-semibold px-2 py-1 rounded-full ${
                  selectedSession.status === 'not_started'
                    ? 'bg-gray-100 text-gray-600'
                    : selectedSession.status === 'in_progress'
                      ? 'bg-amber-100 text-amber-700'
                      : 'bg-emerald-100 text-emerald-700'
                }`}
              >
                {overallLabel}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-4">
              <div className="md:col-span-2">
                <p className="text-xs text-gray-500 mb-1">Case metadata</p>
                <div className="space-y-1 text-sm">
                  <p><span className="font-semibold text-gray-500">Case ID:</span> {selectedSession.caseId}</p>
                  <p><span className="font-semibold text-gray-500">Physician ID:</span> {selectedSession.physicianId}</p>
                  <p><span className="font-semibold text-gray-500">Recorder ID:</span> {selectedSession.recorderId}</p>
                  <p><span className="font-semibold text-gray-500">Case order:</span> {selectedSession.caseOrder ?? '—'}</p>
                </div>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Patient in / Prep start</p>
                <p className="font-mono font-semibold">{formatDate(selectedSession.startedAt)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Vascular access started</p>
                <p className="font-mono font-semibold">{formatDate(selectedSession.accessStartedAt)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Overall prep timer</p>
                <p className="font-mono text-xl font-bold text-[#1e40af]">
                  {formatClock(selectedRuntime.session.totalPrepDurationSec)}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
              <button
                onClick={createNewProcedureSession}
                className="flex items-center justify-center gap-2 px-3 py-3 rounded-xl border border-cyan-300 text-cyan-700 bg-cyan-50 hover:bg-cyan-100 font-semibold text-sm min-h-12"
              >
                <Plus size={16} />
                Start new procedure
              </button>
              <button
                onClick={startSession}
                disabled={!canStartSession}
                className={`flex items-center justify-center gap-2 px-3 py-3 rounded-xl border font-semibold text-sm min-h-12 ${
                  canStartSession
                    ? 'bg-emerald-50 border-emerald-300 text-emerald-700 hover:bg-emerald-100'
                    : 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed'
                }`}
              >
                <Play size={16} />
                Start prep / Patient in
              </button>
              <button
                onClick={markAccessStarted}
                disabled={!canAccess}
                className={`flex items-center justify-center gap-2 px-3 py-3 rounded-xl border font-semibold text-sm min-h-12 ${
                  canAccess
                    ? 'bg-blue-50 border-blue-300 text-blue-700 hover:bg-blue-100'
                    : 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed'
                }`}
              >
                <Timer size={16} />
                Mark vascular access started
              </button>
              <button
                onClick={saveSession}
                disabled={!canSave}
                className={`flex items-center justify-center gap-2 px-3 py-3 rounded-xl border font-semibold text-sm min-h-12 ${
                  canSave
                    ? 'bg-violet-50 border-violet-300 text-violet-700 hover:bg-violet-100'
                    : 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed'
                }`}
              >
                <Save size={16} />
                Save / complete session
              </button>
              <button
                onClick={exportCurrentSession}
                className="flex items-center justify-center gap-2 px-3 py-3 rounded-xl border border-gray-200 text-sm min-h-12 bg-white hover:bg-gray-50 text-gray-700"
              >
                <Download size={16} />
                Export structured JSON
              </button>
            </div>
          </motion.div>

          {demoEditMode && (
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Demo timestamps (optional)</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <label className="text-xs text-gray-600">
                  Session start (patient in)
                  <input
                    type="datetime-local"
                    className="mt-1 w-full rounded-lg border border-gray-300 px-2 py-2 text-sm"
                    value={toDateTimeInput(selectedRuntime.session.startedAt)}
                    onChange={(event) => updateSessionStart(event.target.value)}
                  />
                </label>
                <label className="text-xs text-gray-600">
                  Access started
                  <input
                    type="datetime-local"
                    className="mt-1 w-full rounded-lg border border-gray-300 px-2 py-2 text-sm"
                    value={toDateTimeInput(selectedRuntime.session.accessStartedAt)}
                    onChange={(event) => updateSessionAccess(event.target.value)}
                  />
                </label>
              </div>
            </div>
          )}

          <div className="space-y-4">
            {activeSessionSummary.orderedPhaseRuntimes.map(({ phaseConfig, runtime }) => {
              const needsReasonPrompt = runtime.totalDurationSec >= PREP_NOTES_THRESHOLD_SEC && runtime.phase.reasonCodes.length === 0;
              const reasonLabelMap = new Set(runtime.phase.reasonCodes);

              return (
                <motion.div
                  key={phaseConfig.key}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white rounded-xl border border-gray-200 p-4"
                >
                  <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-2 mb-3">
                    <div>
                      <h3 className="text-base font-semibold text-gray-900">{phaseConfig.label}</h3>
                      <p className="text-xs text-gray-500 mt-1">{phaseConfig.helperText}</p>
                    </div>
                    <span
                      className={`self-start text-[11px] font-semibold px-2 py-1 rounded-full ${
                        runtime.status === 'active'
                          ? 'bg-amber-100 text-amber-700'
                          : runtime.status === 'completed'
                            ? 'bg-emerald-100 text-emerald-700'
                            : 'bg-gray-100 text-gray-500'
                      }`}
                    >
                      {runtime.status === 'not_started' ? 'Not started' : runtime.status === 'active' ? 'Active' : 'Completed'}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
                    <button
                      disabled={!canEditPhases}
                      onClick={() => startPhase(runtime.phase)}
                      className={`min-h-12 rounded-lg border border-emerald-200 text-emerald-700 bg-emerald-50 hover:bg-emerald-100 disabled:bg-gray-100 disabled:text-gray-400 disabled:border-gray-200`}
                    >
                      <div className="flex items-center justify-center gap-2 py-2.5">
                        <Play size={16} />
                        Start
                      </div>
                    </button>
                    <button
                      disabled={!canEditPhases || !runtime.isActive}
                      onClick={() => stopPhase(runtime.phase)}
                      className={`min-h-12 rounded-lg border border-rose-200 text-rose-700 bg-rose-50 hover:bg-rose-100 disabled:bg-gray-100 disabled:text-gray-400 disabled:border-gray-200`}
                    >
                      <div className="flex items-center justify-center gap-2 py-2.5">
                        <StopCircle size={16} />
                        Stop
                      </div>
                    </button>
                    <button
                      onClick={() => toggleExtraordinary(runtime.phase)}
                      className={`min-h-12 rounded-lg border text-sm font-medium ${
                        runtime.phase.extraordinaryFlag
                          ? 'bg-amber-50 text-amber-800 border-amber-300'
                          : 'bg-white text-gray-600 border-gray-200'
                      }`}
                    >
                      <div className="flex items-center justify-center gap-2 py-2.5">
                        <AlertTriangle size={15} />
                        {runtime.phase.extraordinaryFlag ? 'Extraordinary flagged' : 'Mark exceptional'}
                      </div>
                    </button>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-sm text-gray-600 mb-3">
                    <p><span className="font-medium text-gray-500">Elapsed:</span> {formatClock(runtime.totalDurationSec)}</p>
                    <p><span className="font-medium text-gray-500">Start:</span> {formatTimeOnly(runtime.firstStartAt)}</p>
                    <p><span className="font-medium text-gray-500">Last end:</span> {formatTimeOnly(runtime.lastEndAt)}</p>
                    <p><span className="font-medium text-gray-500">Recorder:</span> {runtime.phase.recorderId}</p>
                  </div>

                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 mb-3">
                    <div>
                      <p className="text-xs text-gray-500 mb-2">Primary responsible role</p>
                      <div className="flex flex-wrap gap-2">
                        {roles.map((role) => (
                          <button
                            key={role.id}
                            onClick={() => updatePhasePrimaryRole(runtime.phase, role.id)}
                            className={`px-3 py-2 rounded-full text-xs font-medium border cursor-pointer ${
                              role.id === runtime.phase.primaryRoleId
                                ? 'bg-[#1e40af] text-white border-[#1e40af]'
                                : 'bg-white text-gray-600 border-gray-200'
                            }`}
                          >
                            {role.label}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 mb-2">Supporting roles</p>
                      <div className="flex flex-wrap gap-2">
                        {roles.map((role) => (
                          <button
                            key={role.id}
                            onClick={() => toggleSupportingRole(runtime.phase, role.id)}
                            className={`px-3 py-2 rounded-full text-xs font-medium border cursor-pointer ${
                              runtime.phase.supportingRoleIds.includes(role.id)
                                ? 'bg-gray-800 text-white border-gray-800'
                                : 'bg-gray-100 text-gray-700 border-gray-200'
                            }`}
                          >
                            {role.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="mb-3">
                    <p className="text-xs text-gray-500 mb-2">Responsible staff IDs</p>
                    <div className="flex flex-wrap gap-2">
                      {staff.map((member) => (
                        <button
                          key={member.id}
                          onClick={() => toggleResponsibleStaff(runtime.phase, member.id)}
                          className={`px-3 py-2 rounded-full text-xs font-medium border cursor-pointer ${
                            runtime.phase.responsibleStaffIds.includes(member.id)
                              ? 'bg-[#1e40af]/90 text-white border-[#1e40af]'
                              : 'bg-white text-gray-700 border-gray-200'
                          }`}
                        >
                          {member.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="mb-3">
                    <p className="text-xs text-gray-500 mb-2">Reason codes (quick note)</p>
                    <div className="flex flex-wrap gap-2">
                      {reasonCodes.map((reason) => (
                        <button
                          key={reason.id}
                          onClick={() => toggleReasonCode(runtime.phase, reason.id)}
                          className={`text-left px-3 py-2 rounded-lg border text-xs cursor-pointer ${
                            reasonLabelMap.has(reason.id)
                              ? 'bg-amber-50 text-amber-800 border-amber-300'
                              : 'bg-white text-gray-700 border-gray-200'
                          }`}
                        >
                          {reason.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="text-xs text-gray-500 block mb-2" htmlFor={`note-${runtime.phase.id}`}>
                      Free-text note (optional)
                    </label>
                    <textarea
                      id={`note-${runtime.phase.id}`}
                      value={noteDrafts[runtime.phase.id] ?? ''}
                      onChange={(event) =>
                        setNoteDrafts((prev) => ({ ...prev, [runtime.phase.id]: event.target.value }))
                      }
                      onBlur={() => commitNote(runtime.phase)}
                      disabled={selectedSession.status === 'completed' && !demoEditMode}
                      className="w-full min-h-20 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#1e40af]/30 disabled:bg-gray-100"
                      placeholder="Add optional context for this phase..."
                    />
                  </div>

                  {needsReasonPrompt && (
                    <p className="mt-2 text-xs font-semibold text-amber-700">
                      This phase has exceeded {Math.floor(PREP_NOTES_THRESHOLD_SEC / 60)} minutes. Add a reason chip for faster analysis.
                    </p>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        <div className="xl:col-span-1 space-y-4">
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-gray-900">Session summary</h3>
              <Users size={18} className="text-gray-400" />
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">Status</span>
                <span className="font-semibold">{overallLabel}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Active phases</span>
                <span className="font-semibold">{activeSessionSummary.activePhases.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Phase notes entered</span>
                <span className="font-semibold">{activeSessionSummary.phaseNotes.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Event log entries</span>
                <span className="font-semibold">{selectedSession.eventLog.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Recorded segments</span>
                <span className="font-semibold">
                  {selectedSession.phases.reduce((acc, phase) => acc + phase.segments.length, 0)}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-gray-900">Active phases</h3>
              <Timer size={18} className="text-gray-400" />
            </div>
            <div className="flex flex-wrap gap-2">
              {activeSessionSummary.activePhases.length === 0 ? (
                <p className="text-sm text-gray-500">No currently active phase timers.</p>
              ) : (
                activeSessionSummary.activePhases.map((item) => (
                  <span
                    key={item.runtime.phase.id}
                    className="text-xs px-2 py-1 rounded-full bg-amber-100 text-amber-800 border border-amber-200"
                  >
                    {item.phaseConfig.label}
                  </span>
                ))
              )}
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Captured notes</h3>
            <div className="space-y-3 text-sm">
              {activeSessionSummary.phaseNotes.length === 0 ? (
                <p className="text-gray-500">No notes yet.</p>
              ) : (
                activeSessionSummary.phaseNotes.map((phase) => (
                  <div key={phase.id} className="bg-gray-50 border border-gray-100 rounded-lg p-2">
                    <p className="text-xs font-medium text-gray-500">
                      {PREP_PHASE_CATALOG.find((p) => p.key === phase.key)?.label ?? phase.key}
                    </p>
                    <p className="text-gray-700 text-xs mt-1">
                      {phase.reasonCodes.map((reason) => {
                        const label = reasonCodes.find((r) => r.id === reason)?.label;
                        return label ? `${label}` : reason;
                      }).join(', ')}
                    </p>
                    {phase.freeTextNote && <p className="text-xs text-gray-900 mt-1">{phase.freeTextNote}</p>}
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Recent activity log</h3>
            <div className="space-y-2 text-xs text-gray-600">
              {selectedSession.eventLog.length === 0 && (
                <p className="text-gray-500">No events yet.</p>
              )}
              {[...selectedSession.eventLog].slice(-12).reverse().map((entry) => (
                <div key={entry.id} className="border-l-2 border-gray-200 pl-3 py-1">
                  <p className="font-medium text-gray-700">{buildEventLogLine(entry)}</p>
                  <p className="text-[11px] text-gray-500">{entry.detail ?? ''}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
            <h3 className="font-semibold text-blue-800 mb-2">Preview export payload</h3>
            <pre className="text-[11px] text-gray-700 max-h-52 overflow-auto bg-white p-2 rounded border border-blue-100">
              {JSON.stringify({
                caseId: selectedSession.caseId,
                status: selectedSession.status,
                startedAt: selectedSession.startedAt,
                accessStartedAt: selectedSession.accessStartedAt,
                totalPrepDurationSec: selectedRuntime.session.totalPrepDurationSec,
                phaseDurationsSec: selectedSession.phases.map((phase) => ({
                  phaseKey: phase.key,
                  totalSec: derivePhaseRuntime(phase, tickMs).totalDurationSec,
                  reasonCodes: phase.reasonCodes,
                  extraordinaryFlag: phase.extraordinaryFlag,
                })),
              }, null, 2)}
            </pre>
          </div>
        </div>
      </div>

      <div className="mt-6">
        <button
          onClick={resetToDemo}
          className="flex items-center gap-2 px-3 py-2 rounded-lg border border-amber-300 bg-amber-50 text-amber-800 min-h-10"
        >
          <RotateCcw size={15} />
          Reset demo session(s)
        </button>
      </div>
    </div>
  );
}
