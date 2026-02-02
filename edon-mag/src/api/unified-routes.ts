/**
 * Unified MAG API Routes
 * 
 * Combines Authority, Orchestration, and Explanation layers into single service.
 * All routes use /mag/* prefix (not /mago/*).
 */

import { Router, Request, Response } from "express";
import { MAGAuthorityEngine } from "../core/authority-engine";
import { LedgerBackend } from "../../shared/ledger/types";
import { UniversalIntent } from "../../shared/schemas/uimf";
import { AuthorizationContext, AuthorizationDecision } from "../../shared/schemas/authority";
import { IntentTranslator } from "../orchestration/intent-translator";
import { TaskPlanner } from "../orchestration/task-planner";
import { DataFusionEngine } from "../fusion/data-fusion";
import { Explainer } from "../explanation/explainer";
import { AdapterRegistry } from "../orchestration/adapter-registry";
import { ConstraintApplier } from "../orchestration/constraint-applier";

export function createUnifiedMAGRoutes(
  authorityEngine: MAGAuthorityEngine,
  ledger: LedgerBackend,
  intentTranslator?: IntentTranslator,
  taskPlanner?: TaskPlanner,
  fusionEngine?: DataFusionEngine,
  explainer?: Explainer,
  adapterRegistry?: AdapterRegistry
): Router {
  const router = Router();
  const constraintApplier = new ConstraintApplier();

  // ============================================================================
  // AUTHORITY LAYER ROUTES
  // ============================================================================

  /**
   * POST /mag/authorize
   * Authorize an intent (Authority Layer)
   */
  router.post("/authorize", async (req: Request, res: Response) => {
    try {
      const intent: UniversalIntent = req.body.intent;
      const context: AuthorizationContext = req.body.context || {
        session_id: req.body.session_id || "default",
        operator_id: req.body.operator_id
      };

      // Authorize
      const decision = await authorityEngine.authorize(intent, context);

      // Record in ledger
      await ledger.append({
        event_id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        event_type: decision.decision === "deny" ? "intent_denied" : "intent_authorized",
        payload: {
          intent_id: intent.intent_id,
          decision_id: decision.decision_id,
          decision: decision.decision,
          rationale: decision.rationale,
          risk_level: decision.risk_level,
          constraints: decision.constraints
        },
        signatures: {
          intent_hash: hashIntent(intent),
          decision_hash: hashDecision(decision),
          chain_hash: ""
        },
        metadata: {
          operator_id: context.operator_id,
          session_id: context.session_id,
          mission_id: context.mission_id
        }
      });

      res.json({
        ok: true,
        decision
      });
    } catch (error: any) {
      console.error(`Authorization endpoint error:`, error);
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * POST /mag/assess-risk
   * Assess risk for an intent (Authority Layer)
   */
  router.post("/assess-risk", async (req: Request, res: Response) => {
    try {
      const intent: UniversalIntent = req.body.intent;
      const context: AuthorizationContext = req.body.context || {
        session_id: req.body.session_id || "default"
      };

      const riskAssessment = await authorityEngine.assessRisk(intent, context);

      res.json({
        ok: true,
        risk_assessment: riskAssessment
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * GET /mag/policies
   * List active policy rules (Authority Layer)
   */
  router.get("/policies", async (req: Request, res: Response) => {
    try {
      res.json({
        ok: true,
        policies: [] // Would be populated from policy engine
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * POST /mag/revoke
   * Revoke an authorized intent (Authority Layer)
   */
  router.post("/revoke", async (req: Request, res: Response) => {
    try {
      const { session_id, intent_id, reason, severity } = req.body;
      
      if (!session_id || !intent_id) {
        return res.status(400).json({
          ok: false,
          error: "Missing required fields: session_id, intent_id"
        });
      }
      
      const decisionId = `revoke_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const timestamp = new Date().toISOString();
      const replacementAction = "return_to_safe";
      const rationale = reason || `Intent revoked: ${intent_id}`;
      
      // Log to ledger
      await ledger.append({
        event_id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp,
        event_type: "intent_revoked",
        payload: {
          session_id,
          intent_id,
          decision_id: decisionId,
          reason: rationale,
          replacement_action: replacementAction,
          severity: severity || "high"
        },
        signatures: {
          intent_hash: "",
          decision_hash: "",
          chain_hash: ""
        }
      });
      
      res.json({
        ok: true,
        decision: "revoke",
        replacement_action: replacementAction,
        rationale,
        decision_id: decisionId,
        timestamp
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * GET /mag/ledger/events
   * Query ledger events (Authority Layer)
   */
  router.get("/ledger/events", async (req: Request, res: Response) => {
    try {
      const filters = {
        event_type: req.query.event_type as string | undefined,
        session_id: req.query.session_id as string | undefined,
        mission_id: req.query.mission_id as string | undefined,
        operator_id: req.query.operator_id as string | undefined,
        since: req.query.since as string | undefined,
        until: req.query.until as string | undefined,
        limit: req.query.limit ? parseInt(req.query.limit as string) : undefined,
        offset: req.query.offset ? parseInt(req.query.offset as string) : undefined
      };
      
      const events = await ledger.getEvents(filters);
      
      res.json({
        ok: true,
        events
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * GET /mag/ledger/decisions/{decision_id}
   * Fetch a decision bundle by decision_id (Authority Layer)
   */
  router.get("/ledger/decisions/:decision_id", async (req: Request, res: Response) => {
    try {
      const { decision_id } = req.params;
      if (!decision_id) {
        return res.status(400).json({
          ok: false,
          error: "Missing decision_id"
        });
      }

      const events = await ledger.getEvents({
        limit: 500
      });

      const decisionEvent = events.find((e: any) =>
        e.payload?.decision_id === decision_id
      );

      if (!decisionEvent) {
        return res.status(404).json({
          ok: false,
          error: `Decision not found: ${decision_id}`
        });
      }

      res.json({
        ok: true,
        decision: {
          decision_id,
          decision: decisionEvent.payload?.decision,
          rationale: decisionEvent.payload?.rationale,
          risk_level: decisionEvent.payload?.risk_level,
          constraints: decisionEvent.payload?.constraints,
          intent_id: decisionEvent.payload?.intent_id,
          event_type: decisionEvent.event_type,
          timestamp: decisionEvent.timestamp
        }
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  // ============================================================================
  // ORCHESTRATION LAYER ROUTES
  // ============================================================================

  /**
   * POST /mag/intent
   * Translate natural language to Universal Intent (Orchestration Layer)
   */
  router.post("/intent", async (req: Request, res: Response) => {
    if (!intentTranslator) {
      return res.status(503).json({
        ok: false,
        error: "Orchestration layer not enabled"
      });
    }

    try {
      const { input, source = "text", context, authorize = false } = req.body;
      
      if (!input) {
        return res.status(400).json({
          ok: false,
          error: "Missing 'input' field"
        });
      }
      
      const translationContext = {
        session_id: context?.session_id || req.body.session_id,
        operator_id: context?.operator_id || req.body.operator_id,
        fleet_state: context?.fleet_state || (fusionEngine ? fusionEngine.getFleetState() : undefined),
        domain: context?.domain || req.body.domain,
        mission_mode: context?.mission_mode || req.body.mission_mode
      };
      
      // Translate intent
      const intent = await intentTranslator.translate(
        input,
        source,
        translationContext
      );
      
      // Validate
      const validation = intentTranslator.validate(intent);
      if (!validation.valid) {
        return res.status(400).json({
          ok: false,
          error: "Intent validation failed",
          details: validation.errors
        });
      }
      
      // Optionally authorize with Authority layer
      let authorizationDecision = null;
      if (authorize) {
        try {
          const authContext: AuthorizationContext = {
            session_id: translationContext.session_id || "default",
            operator_id: translationContext.operator_id,
            mission_id: context?.mission_id,
            fleet_state: translationContext.fleet_state,
            mission_mode: translationContext.mission_mode,
            no_go_zones: context?.no_go_zones
          };
          
          authorizationDecision = await authorityEngine.authorize(intent, authContext);
          
          // Record in ledger
          await ledger.append({
            event_id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date().toISOString(),
            event_type: authorizationDecision.decision === "deny" ? "intent_denied" : "intent_authorized",
            payload: {
              intent_id: intent.intent_id,
              decision_id: authorizationDecision.decision_id,
              decision: authorizationDecision.decision,
              rationale: authorizationDecision.rationale,
              risk_level: authorizationDecision.risk_level,
              constraints: authorizationDecision.constraints
            },
            signatures: {
              intent_hash: hashIntent(intent),
              decision_hash: hashDecision(authorizationDecision),
              chain_hash: ""
            },
            metadata: {
              operator_id: authContext.operator_id,
              session_id: authContext.session_id,
              mission_id: authContext.mission_id
            }
          });
          
          // If denied, return immediately
          if (authorizationDecision.decision === "deny") {
            return res.json({
              ok: true,
              intent,
              authorization: authorizationDecision,
              command: null
            });
          }
        } catch (error: any) {
          return res.status(500).json({
            ok: false,
            error: `MAG authorization failed: ${error.message}`
          });
        }
      }
      
      res.json({
        ok: true,
        intent,
        validation,
        authorization: authorizationDecision
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * POST /mag/plan
   * Create task plan from intent (Orchestration Layer)
   */
  router.post("/plan", async (req: Request, res: Response) => {
    if (!taskPlanner || !fusionEngine) {
      return res.status(503).json({
        ok: false,
        error: "Orchestration layer not enabled"
      });
    }

    try {
      const { intent_id, intent, constraints = [] } = req.body;
      
      if (!intent && !intent_id) {
        return res.status(400).json({
          ok: false,
          error: "Missing 'intent' or 'intent_id' field"
        });
      }
      
      // For now, require intent in body
      // In production, would fetch from ledger using intent_id
      if (!intent) {
        return res.status(400).json({
          ok: false,
          error: "Missing 'intent' field (intent_id lookup not yet implemented)"
        });
      }
      
      const fleetState = fusionEngine.getFleetState();
      const taskPlan = taskPlanner.plan(
        intent.structured_intent,
        fleetState,
        constraints
      );
      
      res.json({
        ok: true,
        plan_id: `plan_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        proposed_steps: taskPlan.tasks, // NON-EXECUTABLE
        dependencies: taskPlan.dependencies,
        estimated_duration: taskPlan.estimated_duration
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * GET /mag/fleet-state
   * Get current fleet state (Orchestration Layer)
   */
  router.get("/fleet-state", (req: Request, res: Response) => {
    if (!fusionEngine) {
      return res.status(503).json({
        ok: false,
        error: "Orchestration layer not enabled"
      });
    }

    try {
      const fleetState = fusionEngine.getFleetState();
      res.json({
        ok: true,
        fleet_state: fleetState
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * POST /mag/fusion/ingest
   * Ingest data from external source (Orchestration Layer)
   */
  router.post("/fusion/ingest", (req: Request, res: Response) => {
    if (!fusionEngine) {
      return res.status(503).json({
        ok: false,
        error: "Orchestration layer not enabled"
      });
    }

    try {
      const { source, data, timestamp } = req.body;
      
      if (!source || !data || !timestamp) {
        return res.status(400).json({
          ok: false,
          error: "Missing required fields: source, data, timestamp"
        });
      }
      
      fusionEngine.ingest(source, data, timestamp);
      
      res.json({
        ok: true,
        message: "Data ingested successfully"
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * POST /mag/adapters/{platform}/execute
   * Translate authorized intent to playbook command (Orchestration Layer)
   */
  router.post("/adapters/:platform/execute", async (req: Request, res: Response) => {
    if (!adapterRegistry || !fusionEngine) {
      return res.status(503).json({
        ok: false,
        error: "Orchestration layer not enabled"
      });
    }

    try {
      const platformId = req.params.platform;
      const { intent, skip_authorization = false } = req.body;

      if (!intent) {
        return res.status(400).json({
          ok: false,
          error: "Missing 'intent' field"
        });
      }

      const adapter = adapterRegistry.get(platformId);
      if (!adapter) {
        return res.status(404).json({
          ok: false,
          error: `Adapter not found for platform: ${platformId}`,
          available_platforms: adapterRegistry.listPlatforms()
        });
      }

      let authorizationDecision: AuthorizationDecision | null = null;
      if (!skip_authorization) {
        try {
          const authContext: AuthorizationContext = {
            session_id: intent.context?.session_id || "default",
            operator_id: intent.operator_id,
            mission_id: intent.context?.mission_id,
            fleet_state: intent.context?.fleet_state || fusionEngine.getFleetState(),
            mission_mode: intent.context?.mission_mode,
            no_go_zones: intent.context?.no_go_zones
          };

          authorizationDecision = await authorityEngine.authorize(intent, authContext);

          await ledger.append({
            event_id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date().toISOString(),
            event_type: authorizationDecision.decision === "deny" ? "intent_denied" : "intent_authorized",
            payload: {
              intent_id: intent.intent_id,
              decision_id: authorizationDecision.decision_id,
              decision: authorizationDecision.decision,
              rationale: authorizationDecision.rationale,
              risk_level: authorizationDecision.risk_level,
              constraints: authorizationDecision.constraints
            },
            signatures: {
              intent_hash: hashIntent(intent),
              decision_hash: hashDecision(authorizationDecision),
              chain_hash: ""
            },
            metadata: {
              operator_id: authContext.operator_id,
              session_id: authContext.session_id,
              mission_id: authContext.mission_id
            }
          });

          if (authorizationDecision.decision === "deny") {
            return res.status(403).json({
              ok: false,
              error: "Intent authorization denied",
              decision: authorizationDecision
            });
          }
        } catch (error: any) {
          return res.status(500).json({
            ok: false,
            error: `MAG authorization failed: ${error.message}`
          });
        }
      }

      const proposedCommand = adapter.translate(
        intent.structured_intent,
        {
          ...intent.context,
          intent_id: intent.intent_id,
          fleet_state: intent.context?.fleet_state || fusionEngine.getFleetState()
        }
      );

      const validation = adapter.validate(proposedCommand);
      if (!validation.valid) {
        return res.status(400).json({
          ok: false,
          error: "Command validation failed",
          details: validation.errors
        });
      }

      let finalCommand = proposedCommand;
      let commandModification = null;

      if (authorizationDecision && authorizationDecision.decision === "deny") {
        finalCommand = null;
        authorizationDecision.decision = "deny";
        commandModification = null;
      } else if (
        authorizationDecision &&
        (authorizationDecision.constraints_applied || authorizationDecision.constraints) &&
        (authorizationDecision.constraints_applied || authorizationDecision.constraints)!.length > 0
      ) {
        const constraintContext = {
          mission_mode: intent.context?.mission_mode || req.body.mission_mode || "permissive",
          no_go_zones: intent.context?.no_go_zones || req.body.no_go_zones || []
        };

        commandModification = constraintApplier.applyConstraints(
          proposedCommand,
          authorizationDecision,
          intent.structured_intent,
          constraintContext
        );

        finalCommand = commandModification.final_command;
      } else if (authorizationDecision && authorizationDecision.decision === "degrade") {
        const constraintContext = {
          mission_mode: intent.context?.mission_mode || req.body.mission_mode || "permissive",
          no_go_zones: intent.context?.no_go_zones || req.body.no_go_zones || []
        };

        commandModification = constraintApplier.applyConstraints(
          proposedCommand,
          authorizationDecision,
          intent.structured_intent,
          constraintContext
        );

        finalCommand = commandModification.final_command;
      }

      if (finalCommand === null && authorizationDecision) {
        authorizationDecision.decision = "deny";
        commandModification = null;
      }

      if (!finalCommand && authorizationDecision) {
        authorizationDecision.decision = "deny";
      }

      res.json({
        ok: true,
        command: finalCommand,
        validation,
        authorization: authorizationDecision,
        command_modification: commandModification ? {
          decision: authorizationDecision?.decision || "unknown",
          proposed_command: commandModification.proposed_command,
          final_command: commandModification.final_command,
          policy_triggered: commandModification.policy_triggered,
          modifications: commandModification.modifications,
          reason: commandModification.reason
        } : null
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  // ============================================================================
  // EXPLANATION LAYER ROUTES
  // ============================================================================

  /**
   * GET /mag/decisions/{decision_id}/explain
   * Explain decision from ledger (Explanation Layer)
   */
  router.get("/decisions/:decision_id/explain", async (req: Request, res: Response) => {
    if (!explainer) {
      return res.status(503).json({
        ok: false,
        error: "Explanation layer not enabled"
      });
    }

    try {
      const { decision_id } = req.params;
      
      // Query ledger for decision
      const events = await ledger.getEvents({
        event_type: "intent_authorized",
        limit: 100
      });
      
      // Find decision in ledger
      const decisionEvent = events.find((e: any) => 
        e.payload?.decision_id === decision_id
      );
      
      if (!decisionEvent) {
        return res.status(404).json({
          ok: false,
          error: `Decision not found: ${decision_id}`
        });
      }
      
      // For now, require intent and decision in query params
      // In production, would reconstruct from ledger
      const { intent, decision } = req.query;
      
      if (!intent || !decision) {
        return res.status(400).json({
          ok: false,
          error: "Missing 'intent' or 'decision' in query params (ledger reconstruction not yet implemented)"
        });
      }
      
      const explanation = explainer.explain(
        JSON.parse(intent as string),
        JSON.parse(decision as string),
        {
          session_id: decisionEvent.metadata?.session_id,
          mission_id: decisionEvent.metadata?.mission_id,
          operator_id: decisionEvent.metadata?.operator_id
        }
      );
      
      res.json({
        ok: true,
        decision_id,
        explanation,
        policy_version: "1.0", // Would come from ledger
        original_intent: JSON.parse(intent as string)
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  /**
   * POST /mag/explain
   * Get plain-language explanation (Explanation Layer - legacy endpoint)
   */
  router.post("/explain", (req: Request, res: Response) => {
    if (!explainer) {
      return res.status(503).json({
        ok: false,
        error: "Explanation layer not enabled"
      });
    }

    try {
      const { intent_id, decision_id, intent, decision, context } = req.body;
      
      if (!intent || !decision) {
        return res.status(400).json({
          ok: false,
          error: "Missing 'intent' or 'decision' field"
        });
      }
      
      const explanation = explainer.explain(
        intent,
        decision,
        context || {}
      );
      
      res.json({
        ok: true,
        explanation
      });
    } catch (error: any) {
      res.status(500).json({
        ok: false,
        error: error.message
      });
    }
  });

  return router;
}

function hashIntent(intent: UniversalIntent): string {
  return Buffer.from(JSON.stringify(intent)).toString("base64").substr(0, 32);
}

function hashDecision(decision: AuthorizationDecision): string {
  return Buffer.from(JSON.stringify(decision)).toString("base64").substr(0, 32);
}
