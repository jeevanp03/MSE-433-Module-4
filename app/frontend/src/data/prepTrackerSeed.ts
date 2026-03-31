import type {
  PrepEventLogEntry,
  PrepPhase,
  PrepSession,
  PrepRole,
  PrepPhaseTemplate,
  PrepReasonCode,
  PrepStaff,
  PrepTrackerSeedData,
} from '../types/prepTracker';

const NOTE_PROMPT_THRESHOLD_SEC = 180;

const ROLE_CATALOG: PrepRole[] = [
  { id: 'anaesthesia', label: 'Anaesthesia' },
  { id: 'ep-nurse', label: 'EP Nurse / Scrub Nurse' },
  { id: 'circulating-nurse', label: 'Circulating Nurse' },
  { id: 'physician', label: 'Physician' },
  { id: 'mapping-tech', label: 'Mapping Technologist' },
  { id: 'other', label: 'Other' },
];

const STAFF_ROSTER: PrepStaff[] = [
  { id: 'RN-01', label: 'RN-01', roleHint: 'Circulating Nurse' },
  { id: 'RN-02', label: 'RN-02', roleHint: 'Circulating Nurse' },
  { id: 'RN-03', label: 'RN-03', roleHint: 'EP Nurse / Scrub Nurse' },
  { id: 'RN-04', label: 'RN-04', roleHint: 'EP Nurse / Scrub Nurse' },
  { id: 'AN-11', label: 'AN-11', roleHint: 'Anaesthesia' },
  { id: 'MD-07', label: 'MD-07', roleHint: 'Physician' },
  { id: 'MD-12', label: 'MD-12', roleHint: 'Physician' },
  { id: 'TM-08', label: 'TM-08', roleHint: 'Mapping Technologist' },
];

const REASON_CODES: PrepReasonCode[] = [
  { id: 'difficult_airway', label: 'Difficult airway / extended intubation' },
  { id: 'patient_reposition', label: 'Patient repositioning needed' },
  { id: 'monitoring_setup', label: 'Monitoring setup issue' },
  { id: 'equipment_not_ready', label: 'Equipment not ready' },
  { id: 'supply_retrieval', label: 'Supply retrieval / missing items' },
  { id: 'cable_interference', label: 'Cable / line interference' },
  { id: 'waiting_staff', label: 'Waiting on staff / handoff' },
  { id: 'communication_delay', label: 'Communication delay / clarification' },
  { id: 'room_setup_issue', label: 'Room setup issue' },
  { id: 'sterility_issue', label: 'Sterility issue / reset' },
  { id: 'patient_complexity', label: 'Patient-specific complexity' },
  { id: 'safety_pause', label: 'Safety-related pause/check' },
  { id: 'other', label: 'Other' },
];

const PHASE_CATALOG: PrepPhaseTemplate[] = [
  {
    key: 'patient_transfer',
    label: 'Patient transfer & positioning',
    helperText: 'Table transfer, line checks, and positioning for workflow readiness.',
  },
  {
    key: 'monitoring_setup',
    label: 'Monitoring hookup / baseline setup',
    helperText: 'Connect ECG, pulse ox, BP, and anesthesia monitoring baseline.',
  },
  {
    key: 'anesthesia_induction',
    label: 'Anaesthesia induction',
    helperText: 'Premedication and induction handoff steps.',
  },
  {
    key: 'airway',
    label: 'Airway / intubation',
    helperText: 'Laryngoscopy, placement, confirmation, and securement.',
  },
  { key: 'sterile_prep', label: 'Sterile prep', helperText: 'Prep skin, drape prep site boundaries.' },
  {
    key: 'sterile_draping',
    label: 'Sterile draping',
    helperText: 'Barrier placement and sterile field set-up.',
  },
  {
    key: 'equipment_readiness',
    label: 'Equipment / room readiness',
    helperText: 'Parallel checks: leads, catheters, imaging carts, backup supplies.',
  },
  { key: 'final_ready', label: 'Final ready-for-access check', helperText: 'Time-out and procedural start clearance.' },
];

type RelativeSegment = { startMin: number; endMin?: number | null };

interface CaseSeedSpec {
  id: string;
  caseId: string;
  dateOffsetMin: number;
  physicianId: string;
  recorderId: string;
  caseOrder: number;
  status: PrepSession['status'];
  startedAtMin: number;
  accessAtMin?: number | null;
  phaseSegments: Record<string, RelativeSegment[]>;
  phaseDefaults: Record<string, { primaryRoleId: string; supportingRoleIds: string[]; staffIds: string[] }>;
  phaseFlags?: Record<string, { extraordinary?: boolean; reasonCodes?: string[]; freeTextNote?: string }>;
}

let idCounter = 0;
const nowIso = (baseMs: number, minutes: number) =>
  new Date(baseMs + minutes * 60_000).toISOString();

const seedId = (prefix: string) => {
  idCounter += 1;
  return `${prefix}-${idCounter.toString(36)}-${Date.now().toString(36)}`;
};

const segmentFromMinutes = (baseMs: number, def: RelativeSegment) => ({
  id: seedId('seg'),
  startedAt: nowIso(baseMs, def.startMin),
  ...(def.endMin != null ? { endedAt: nowIso(baseMs, def.endMin) } : {}),
});

const derivePhaseStatus = (segments: { endedAt?: string | null; startedAt: string }[]): PrepPhase['status'] => {
  if (segments.length === 0) return 'not_started';
  return segments.some((s) => s.endedAt == null) ? 'active' : 'completed';
};

const buildPhase = (
  key: string,
  baseMs: number,
  segmentDefs: RelativeSegment[],
  recorderId: string,
  defaults: CaseSeedSpec['phaseDefaults'][string],
  flags?: { extraordinary?: boolean; reasonCodes?: string[]; freeTextNote?: string },
): PrepPhase => {
  const segments = segmentDefs.map((seg) => segmentFromMinutes(baseMs, seg));
  return {
    id: seedId(`phase-${key}`),
    key,
    status: derivePhaseStatus(segments),
    primaryRoleId: defaults.primaryRoleId,
    supportingRoleIds: defaults.supportingRoleIds,
    responsibleStaffIds: defaults.staffIds,
    recorderId,
    segments,
    extraordinaryFlag: !!flags?.extraordinary,
    reasonCodes: flags?.reasonCodes ?? [],
    freeTextNote: flags?.freeTextNote ?? '',
  };
};

const buildEvent = (
  idSeed: string,
  timestamp: string,
  action: PrepEventLogEntry['action'],
  recorderId: string,
  phaseId?: string,
  detail?: string,
): PrepEventLogEntry => ({
  id: seedId(idSeed),
  timestamp,
  phaseId,
  action,
  recorderId,
  detail,
});

const eventsFromSegments = (
  segments: Array<{ startedAt: string; endedAt?: string | null }>,
  phaseId: string,
  recorderId: string,
): PrepEventLogEntry[] => {
  const events: PrepEventLogEntry[] = [];
  for (const seg of segments) {
    events.push(buildEvent('ev', seg.startedAt, 'start', recorderId, phaseId));
    if (seg.endedAt) {
      events.push(buildEvent('ev', seg.endedAt, 'stop', recorderId, phaseId));
    }
  }
  return events;
};

const computeTotalSeconds = (startedAt: string, accessAt: string | null): number => {
  if (!startedAt) return 0;
  const startMs = Date.parse(startedAt);
  const endMs = accessAt ? Date.parse(accessAt) : Date.now();
  const delta = endMs - startMs;
  return Math.max(0, Math.floor(delta / 1000));
};

const buildBlankSession = (baseNowMs: number): PrepSession => {
  const caseIdSuffix = (++idCounter).toString().padStart(3, '0');
  const defaultRecorderId = 'RC-01';
  const defaultPhysicianId = 'DR-P2';
  const fallbackRoleId = ROLE_CATALOG[0]?.id ?? 'circulating-nurse';
  const phaseRecorderId = defaultRecorderId;

  return {
    id: `sess-new-${idCounter}-${Date.now().toString(36)}`,
    caseId: `DEMO-AF-NEW-${caseIdSuffix}`,
    dateTime: new Date(baseNowMs).toISOString(),
    physicianId: defaultPhysicianId,
    recorderId: defaultRecorderId,
    startedAt: null,
    accessStartedAt: null,
    status: 'not_started',
    caseOrder: 1,
    phases: PREP_PHASE_CATALOG.map((phase, index) => ({
      id: `phase-${idCounter}-${phase.key}-${index}-${Date.now().toString(36)}`,
      key: phase.key,
      status: 'not_started',
      primaryRoleId: fallbackRoleId,
      supportingRoleIds: [],
      responsibleStaffIds: [],
      recorderId: phaseRecorderId,
      segments: [],
      extraordinaryFlag: false,
      reasonCodes: [],
      freeTextNote: '',
    })),
    eventLog: [],
    totalPrepDurationSec: 0,
  };
};

const buildSession = (baseNow: number, spec: CaseSeedSpec): PrepSession => {
  const caseStartMs = baseNow + spec.dateOffsetMin * 60_000;
  const startedAt = nowIso(caseStartMs, spec.startedAtMin);
  const accessStartedAt = spec.accessAtMin == null ? null : nowIso(caseStartMs, spec.accessAtMin);
  const eventLog: PrepEventLogEntry[] = [];
  const phases: PrepPhase[] = [];
  const allReasons = Object.entries(spec.phaseSegments);

  for (const [phaseKey, segments] of allReasons) {
    const defaults = spec.phaseDefaults[phaseKey];
    if (!defaults) continue;

    const flag = spec.phaseFlags?.[phaseKey];
    const phase = buildPhase(phaseKey, caseStartMs, segments, spec.recorderId, defaults, flag);
    phases.push(phase);
    eventLog.push(
      ...eventsFromSegments(phase.segments, phase.id, phase.recorderId),
    );
  }

  eventLog.push(buildEvent('ev', startedAt, 'session_started', spec.recorderId));
  if (accessStartedAt) {
    eventLog.push(buildEvent('ev', accessStartedAt, 'access_started', spec.recorderId));
  }

  const totalPrepDurationSec = computeTotalSeconds(startedAt, accessStartedAt);

  return {
    id: spec.id,
    caseId: spec.caseId,
    dateTime: startedAt,
    physicianId: spec.physicianId,
    recorderId: spec.recorderId,
    startedAt,
    accessStartedAt,
    status: spec.status,
    caseOrder: spec.caseOrder,
    phases,
    eventLog: eventLog.sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp)),
    totalPrepDurationSec,
  };
};

export const createSeedTrackerState = (referenceNowMs = Date.now()) => {
  const activeNowMinutes = -16;
  const completedCaseAOffset = -3 * 24 * 60;
  const completedCaseBOffset = -1 * 24 * 60;
  const completedCaseCOffset = -2 * 24 * 60 - 180;
  const completedCaseDOffset = -12 * 24 * 60 - 120;
  const completedCaseEOffset = -5 * 24 * 60 - 90;

  const sessionSpecs: CaseSeedSpec[] = [
    {
      id: 'sess-active-01',
      caseId: 'DEMO-AF-102',
      dateOffsetMin: activeNowMinutes,
      physicianId: 'DR-P2',
      recorderId: 'RC-01',
      caseOrder: 17,
      status: 'completed',
      startedAtMin: 0,
      accessAtMin: 40,
      phaseSegments: {
        patient_transfer: [{ startMin: 0, endMin: 4 }],
        monitoring_setup: [{ startMin: 1, endMin: 8 }],
        anesthesia_induction: [{ startMin: 8, endMin: 16 }],
        airway: [{ startMin: 12, endMin: 24 }],
        sterile_prep: [{ startMin: 10, endMin: 14 }],
        sterile_draping: [{ startMin: 13, endMin: 17 }],
        equipment_readiness: [{ startMin: 5, endMin: 10 }, { startMin: 12, endMin: 26 }],
        final_ready: [{ startMin: 30, endMin: 40 }],
      },
      phaseDefaults: {
        patient_transfer: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['ep-nurse'], staffIds: ['RN-01', 'RN-03'] },
        monitoring_setup: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11', 'RN-02'] },
        anesthesia_induction: { primaryRoleId: 'anesthesia', supportingRoleIds: ['physician'], staffIds: ['AN-11'] },
        airway: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11', 'RN-02'] },
        sterile_prep: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-03', 'RN-01'] },
        sterile_draping: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-03'] },
        equipment_readiness: { primaryRoleId: 'mapping-tech', supportingRoleIds: ['circulating-nurse'], staffIds: ['TM-08'] },
        final_ready: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['physician'], staffIds: ['RN-01'] },
      },
      phaseFlags: {
        airway: {
          reasonCodes: ['communication_delay'],
          freeTextNote: 'Airway confirmation delayed by equipment positioning overlap with sterile prep.',
        },
      },
    },
    {
      id: 'sess-completed-02',
      caseId: 'DEMO-AF-089',
      dateOffsetMin: completedCaseAOffset + -45,
      physicianId: 'DR-P1',
      recorderId: 'RC-02',
      caseOrder: 12,
      status: 'completed',
      startedAtMin: 0,
      accessAtMin: 45,
      phaseSegments: {
        patient_transfer: [{ startMin: 0, endMin: 4 }],
        monitoring_setup: [{ startMin: 1, endMin: 9 }],
        anesthesia_induction: [{ startMin: 9, endMin: 23 }, { startMin: 24, endMin: 33 }],
        airway: [{ startMin: 33, endMin: 40 }],
        sterile_prep: [{ startMin: 22, endMin: 30 }],
        sterile_draping: [{ startMin: 30, endMin: 38 }],
        equipment_readiness: [{ startMin: 14, endMin: 31 }],
        final_ready: [{ startMin: 38, endMin: 45 }],
      },
      phaseDefaults: {
        patient_transfer: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['ep-nurse'], staffIds: ['RN-01', 'RN-03'] },
        monitoring_setup: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11'] },
        anesthesia_induction: { primaryRoleId: 'anesthesia', supportingRoleIds: ['physician'], staffIds: ['AN-11', 'MD-07'] },
        airway: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11'] },
        sterile_prep: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-03'] },
        sterile_draping: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-03'] },
        equipment_readiness: { primaryRoleId: 'mapping-tech', supportingRoleIds: ['circulating-nurse'], staffIds: ['TM-08'] },
        final_ready: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['physician'], staffIds: ['RN-02'] },
      },
      phaseFlags: {
        anesthesia_induction: {
          extraordinary: true,
          reasonCodes: ['difficult_airway'],
          freeTextNote: 'Second induction attempt after transient hypoventilation; delayed start of airway phase.',
        },
      },
    },
    {
      id: 'sess-completed-03',
      caseId: 'DEMO-AF-088',
      dateOffsetMin: completedCaseBOffset + -95,
      physicianId: 'DR-P3',
      recorderId: 'RC-03',
      caseOrder: 8,
      status: 'completed',
      startedAtMin: 0,
      accessAtMin: 50,
      phaseSegments: {
        patient_transfer: [{ startMin: 0, endMin: 5 }],
        monitoring_setup: [{ startMin: 2, endMin: 10 }],
        anesthesia_induction: [{ startMin: 10, endMin: 18 }],
        airway: [{ startMin: 18, endMin: 24 }],
        sterile_prep: [{ startMin: 20, endMin: 33 }],
        sterile_draping: [{ startMin: 34, endMin: 44 }],
        equipment_readiness: [{ startMin: 10, endMin: 31 }, { startMin: 31, endMin: 37 }],
        final_ready: [{ startMin: 44, endMin: 50 }],
      },
      phaseDefaults: {
        patient_transfer: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['ep-nurse'], staffIds: ['RN-02', 'RN-04'] },
        monitoring_setup: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11', 'RN-02'] },
        anesthesia_induction: { primaryRoleId: 'anesthesia', supportingRoleIds: ['physician'], staffIds: ['AN-11', 'MD-12'] },
        airway: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11'] },
        sterile_prep: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-03', 'RN-04'] },
        sterile_draping: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-03'] },
        equipment_readiness: { primaryRoleId: 'mapping-tech', supportingRoleIds: ['circulating-nurse'], staffIds: ['TM-08'] },
        final_ready: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['physician'], staffIds: ['RN-01', 'MD-12'] },
      },
      phaseFlags: {
        equipment_readiness: {
          extraordinary: true,
          reasonCodes: ['equipment_not_ready', 'supply_retrieval'],
          freeTextNote: 'Backup circuit and extra drape kits were missing on first pass; resolved at 37 minutes.',
        },
      },
    },
    {
      id: 'sess-completed-04',
      caseId: 'DEMO-AF-076',
      dateOffsetMin: completedCaseCOffset,
      physicianId: 'DR-P2',
      recorderId: 'RC-02',
      caseOrder: 5,
      status: 'completed',
      startedAtMin: 0,
      accessAtMin: 46,
      phaseSegments: {
        patient_transfer: [{ startMin: 0, endMin: 4 }],
        monitoring_setup: [{ startMin: 2, endMin: 9 }],
        anesthesia_induction: [{ startMin: 9, endMin: 16 }],
        airway: [{ startMin: 16, endMin: 23 }],
        sterile_prep: [{ startMin: 20, endMin: 28 }],
        sterile_draping: [{ startMin: 28, endMin: 34 }],
        equipment_readiness: [{ startMin: 12, endMin: 25 }],
        final_ready: [{ startMin: 34, endMin: 46 }],
      },
      phaseDefaults: {
        patient_transfer: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['ep-nurse'], staffIds: ['RN-01', 'RN-03'] },
        monitoring_setup: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11', 'RN-01'] },
        anesthesia_induction: { primaryRoleId: 'anesthesia', supportingRoleIds: ['physician'], staffIds: ['AN-11', 'MD-07'] },
        airway: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11'] },
        sterile_prep: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-04'] },
        sterile_draping: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-04'] },
        equipment_readiness: { primaryRoleId: 'mapping-tech', supportingRoleIds: ['circulating-nurse'], staffIds: ['TM-08'] },
        final_ready: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['physician'], staffIds: ['RN-02', 'MD-07'] },
      },
      phaseFlags: {
        monitoring_setup: {
          extraordinary: true,
          reasonCodes: ['waiting_staff'],
          freeTextNote: 'Primary circulating nurse was pulled to assist an emergent handoff.',
        },
      },
    },
    {
      id: 'sess-completed-05',
      caseId: 'DEMO-AF-061',
      dateOffsetMin: completedCaseDOffset + -20,
      physicianId: 'DR-P1',
      recorderId: 'RC-01',
      caseOrder: 3,
      status: 'completed',
      startedAtMin: 0,
      accessAtMin: 34,
      phaseSegments: {
        patient_transfer: [{ startMin: 0, endMin: 3 }],
        monitoring_setup: [{ startMin: 2, endMin: 8 }],
        anesthesia_induction: [{ startMin: 8, endMin: 16 }],
        airway: [{ startMin: 16, endMin: 22 }],
        sterile_prep: [{ startMin: 19, endMin: 26 }],
        sterile_draping: [{ startMin: 26, endMin: 31 }],
        equipment_readiness: [{ startMin: 17, endMin: 30 }],
        final_ready: [{ startMin: 31, endMin: 34 }],
      },
      phaseDefaults: {
        patient_transfer: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['ep-nurse'], staffIds: ['RN-02', 'RN-01'] },
        monitoring_setup: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11'] },
        anesthesia_induction: { primaryRoleId: 'anesthesia', supportingRoleIds: ['physician'], staffIds: ['AN-11', 'MD-07'] },
        airway: { primaryRoleId: 'anesthesia', supportingRoleIds: ['circulating-nurse'], staffIds: ['AN-11'] },
        sterile_prep: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-03'] },
        sterile_draping: { primaryRoleId: 'ep-nurse', supportingRoleIds: ['circulating-nurse'], staffIds: ['RN-03'] },
        equipment_readiness: { primaryRoleId: 'mapping-tech', supportingRoleIds: ['circulating-nurse'], staffIds: ['TM-08'] },
        final_ready: { primaryRoleId: 'circulating-nurse', supportingRoleIds: ['physician'], staffIds: ['RN-01', 'MD-12'] },
      },
    },
  ];

  const sessions = sessionSpecs.map((spec) => buildSession(referenceNowMs, spec));
  const starterSession = buildBlankSession(referenceNowMs);

  return {
    sessions: [starterSession, ...sessions].sort((a, b) => Date.parse(b.dateTime) - Date.parse(a.dateTime)),
    activeSessionId: starterSession.id,
    roles: ROLE_CATALOG,
    staff: STAFF_ROSTER,
    reasonCodes: REASON_CODES,
    phaseCatalog: PHASE_CATALOG,
    notePromptThresholdSec: NOTE_PROMPT_THRESHOLD_SEC,
  };
};

export const PREP_PHASE_CATALOG = PHASE_CATALOG;
export const PREP_ROLES = ROLE_CATALOG;
export const PREP_STAFF = STAFF_ROSTER;
export const PREP_REASON_CODES = REASON_CODES;
export const PREP_NOTES_THRESHOLD_SEC = NOTE_PROMPT_THRESHOLD_SEC;
export const buildBlankPrepSession = buildBlankSession;

export const PREP_TRACKER_SEED: PrepTrackerSeedData = {
  sessions: createSeedTrackerState(Date.now()).sessions,
  roles: ROLE_CATALOG,
  staff: STAFF_ROSTER,
  reasonCodes: REASON_CODES,
  phaseCatalog: PHASE_CATALOG,
  notePromptThresholdSec: NOTE_PROMPT_THRESHOLD_SEC,
};

export const PREP_TRACER_SEED = PREP_TRACKER_SEED;
