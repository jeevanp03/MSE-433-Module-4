export type PrepSessionStatus = 'not_started' | 'in_progress' | 'completed';
export type PrepPhaseStatus = 'not_started' | 'active' | 'completed';
export type PrepEventAction = 'start' | 'stop' | 'note_added' | 'role_changed' | 'session_started' | 'access_started';

export interface PrepPhaseSegment {
  id: string;
  startedAt: string;
  endedAt?: string | null;
}

export interface PrepEventLogEntry {
  id: string;
  timestamp: string;
  action: PrepEventAction;
  phaseId?: string;
  recorderId: string;
  detail?: string;
}

export interface PrepPhase {
  id: string;
  key: string;
  status: PrepPhaseStatus;
  primaryRoleId: string;
  supportingRoleIds: string[];
  responsibleStaffIds: string[];
  recorderId: string;
  segments: PrepPhaseSegment[];
  extraordinaryFlag: boolean;
  reasonCodes: string[];
  freeTextNote: string;
}

export interface PrepSession {
  id: string;
  caseId: string;
  dateTime: string;
  physicianId: string;
  recorderId: string;
  startedAt: string | null;
  accessStartedAt: string | null;
  status: PrepSessionStatus;
  caseOrder?: number;
  phases: PrepPhase[];
  eventLog: PrepEventLogEntry[];
  totalPrepDurationSec: number;
}

export interface PrepPhaseTemplate {
  key: string;
  label: string;
  helperText?: string;
}

export interface PrepReasonCode {
  id: string;
  label: string;
}

export interface PrepRole {
  id: string;
  label: string;
}

export interface PrepStaff {
  id: string;
  label: string;
  roleHint: string;
}

export interface PrepTrackerSeedData {
  sessions: PrepSession[];
  roles: PrepRole[];
  staff: PrepStaff[];
  reasonCodes: PrepReasonCode[];
  phaseCatalog: PrepPhaseTemplate[];
  notePromptThresholdSec: number;
}
