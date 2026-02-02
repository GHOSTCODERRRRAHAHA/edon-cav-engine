/**
 * Intent Translator
 * 
 * Translates voice/text/GUI input to structured Universal Intent.
 */

import { UniversalIntent, StructuredIntent, SourceType } from "../../../shared/schemas/uimf";
import { v4 as uuidv4 } from "uuid";

export interface IntentTranslator {
  translate(
    input: string | Buffer,
    source: SourceType,
    context: TranslationContext
  ): Promise<UniversalIntent>;
  
  validate(intent: UniversalIntent): ValidationResult;
  
  scoreConfidence(intent: UniversalIntent): number;
}

export interface TranslationContext {
  session_id?: string;
  operator_id?: string;
  fleet_state?: any;
  domain?: string;
  mission_mode?: "permissive" | "contested" | "degraded";
}

export interface ValidationResult {
  valid: boolean;
  errors?: string[];
  warnings?: string[];
}

export class LLMIntentTranslator implements IntentTranslator {
  private llmProvider: "openai" | "anthropic" | "local";
  private apiKey?: string;

  constructor(provider: "openai" | "anthropic" | "local" = "openai", apiKey?: string) {
    this.llmProvider = provider;
    this.apiKey = apiKey;
  }

  async translate(
    input: string | Buffer,
    source: SourceType,
    context: TranslationContext
  ): Promise<UniversalIntent> {
    // Convert audio to text if needed
    const textInput = typeof input === "string" ? input : await this.transcribeAudio(input);
    
    // Use LLM to extract structured intent
    const structuredIntent = await this.extractStructuredIntent(textInput, context);
    
    // Build Universal Intent
    const intent: UniversalIntent = {
      intent_id: uuidv4(),
      timestamp: new Date().toISOString(),
      source,
      operator_id: context.operator_id,
      intent_type: this.determineIntentType(structuredIntent),
      natural_language: textInput,
      structured_intent: structuredIntent,
      context: {
        session_id: context.session_id,
        mission_id: context.domain ? `mission_${context.domain}` : undefined,
        fleet_state: context.fleet_state,
        mission_mode: context.mission_mode
      },
      confidence: 0.7, // Will be computed below
      domain: context.domain
    };
    
    // Compute confidence now that intent is complete
    intent.confidence = this.scoreConfidence(intent);
    
    return intent;
  }

  validate(intent: UniversalIntent): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    
    // Validate required fields
    if (!intent.intent_id) {
      errors.push("Missing intent_id");
    }
    if (!intent.structured_intent.action) {
      errors.push("Missing structured_intent.action");
    }
    if (!intent.natural_language) {
      warnings.push("Missing natural_language");
    }
    
    // Validate action is known
    const knownActions = ["shadow", "patrol", "monitor", "return", "engage", "intercept", "hover", "land", "track", "search", "scan", "inspect"];
    if (!knownActions.includes(intent.structured_intent.action)) {
      warnings.push(`Unknown action: ${intent.structured_intent.action}`);
    }
    
    return {
      valid: errors.length === 0,
      errors: errors.length > 0 ? errors : undefined,
      warnings: warnings.length > 0 ? warnings : undefined
    };
  }

  scoreConfidence(intent: UniversalIntent): number {
    // Simple confidence scoring
    // In production, this would use LLM confidence scores
    let confidence = 0.7; // Base confidence
    
    // Check if structured_intent exists
    if (!intent.structured_intent || !intent.structured_intent.action) {
      return confidence;
    }
    
    // Increase confidence if action is well-known
    const knownActions = ["shadow", "patrol", "monitor", "return"];
    if (knownActions.includes(intent.structured_intent.action)) {
      confidence += 0.15;
    }
    
    // Increase confidence if target is specified
    if (intent.structured_intent.target) {
      confidence += 0.1;
    }
    
    // Decrease confidence if natural language is very short
    if (intent.natural_language && intent.natural_language.length < 10) {
      confidence -= 0.1;
    }
    
    return Math.min(Math.max(confidence, 0.0), 1.0);
  }

  private async transcribeAudio(audio: Buffer): Promise<string> {
    // Placeholder - would use actual speech-to-text service
    throw new Error("Audio transcription not yet implemented");
  }

  private async extractStructuredIntent(
    text: string,
    context: TranslationContext
  ): Promise<StructuredIntent> {
    // Placeholder - would use LLM with function calling
    // For now, use simple rule-based parsing
    
    const lowerText = text.toLowerCase();
    
    // Extract action
    let action = "monitor"; // Default
    if (lowerText.includes("shadow") || lowerText.includes("follow")) {
      action = "shadow";
    } else if (lowerText.includes("patrol")) {
      action = "patrol";
    } else if (lowerText.includes("return") || lowerText.includes("home") || lowerText.includes("base")) {
      action = "return";
    } else if (lowerText.includes("hover") || lowerText.includes("hold position")) {
      action = "hover";
    } else if (lowerText.includes("land")) {
      action = "land";
    } else if (lowerText.includes("track")) {
      action = "track";
    } else if (lowerText.includes("search") || lowerText.includes("look for")) {
      action = "search";
    } else if (lowerText.includes("scan")) {
      action = "scan";
    } else if (lowerText.includes("inspect") || lowerText.includes("examine")) {
      action = "inspect";
    }
    
    // Extract target (simplified)
    let target = undefined;
    if (lowerText.includes("vessel") || lowerText.includes("ship")) {
      target = {
        type: "entity" as const,
        identifier: "vessel_1"
      };
    }
    
    // Extract parameters (enhanced)
    const parameters: Record<string, any> = {};
    
    // Distance parsing (improved)
    const distanceMatch = text.match(/(\d+(?:\.\d+)?)\s*(km|kilometers?|miles?|meters?|m|nautical\s*miles?|nm)/i);
    if (distanceMatch) {
      parameters.distance_km = this.parseDistance(distanceMatch[0]);
    }
    
    // Coordinate parsing (lat, lon)
    let coordMatch = text.match(/(?:coordinates?|position|location|at)\s+(?:the\s+)?(?:vessel|ship|target|object)?\s*(?:at\s+)?coordinates?\s*([+-]?\d+\.?\d+)\s*[,;]\s*([+-]?\d+\.?\d+)/i);
    
    // Pattern 2: Just "X, Y" (standalone coordinates)
    if (!coordMatch) {
      coordMatch = text.match(/([+-]?\d+\.\d+)\s*[,;]\s*([+-]?\d+\.\d+)/);
    }
    
    // Pattern 3: "at X, Y" (coordinates after "at")
    if (!coordMatch) {
      coordMatch = text.match(/at\s+([+-]?\d+\.\d+)\s*[,;]\s*([+-]?\d+\.\d+)/i);
    }
    
    if (coordMatch) {
      const lat = parseFloat(coordMatch[1]);
      const lon = parseFloat(coordMatch[2]);
      console.log(`[INTENT] Extracted coordinates: (${lat}, ${lon}) from text: "${text}"`);
      
      parameters.coordinates = {
        lat: lat,
        lon: lon
      };
      // Also set in target if action is inspect/examine
      if (action === "inspect" || action === "examine") {
        if (!target) {
          target = { type: "location" as const };
        }
        if (typeof target === "object" && "type" in target) {
          (target as any).coordinates = {
            lat: lat,
            lon: lon
          };
        }
        console.log(`[INTENT] Set coordinates in target for inspect action: (${lat}, ${lon})`);
      }
    } else {
      console.log(`[INTENT] No coordinates found in text: "${text}"`);
    }
    
    // Time duration parsing
    const timeMatch = text.match(/(?:for|duration|last)\s*(\d+)\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)/i);
    if (timeMatch) {
      parameters.duration_seconds = this.parseDuration(timeMatch[0]);
    }
    
    // Speed parsing
    const speedMatch = text.match(/(?:at|speed|velocity)\s*(\d+)\s*(?:km\/h|kmh|mph|m\/s|knots?|kts?)/i);
    if (speedMatch) {
      parameters.speed_ms = this.parseSpeed(speedMatch[0]);
    }
    
    // Altitude parsing
    const altitudeMatch = text.match(/(?:at|altitude|height)\s*(\d+)\s*(?:meters?|m|feet|ft)/i);
    if (altitudeMatch) {
      parameters.altitude_m = this.parseAltitude(altitudeMatch[0]);
    }
    
    // Radius/area parsing (improved patterns)
    const radiusPatterns = [
      /(\d+(?:\.\d+)?)\s*(?:km|kilometers?)\s*radius/i,  // "5km radius"
      /(\d+(?:\.\d+)?)\s*(?:mile|miles)\s*radius/i,      // "3 mile radius"
      /radius\s*(?:of|at)?\s*(\d+(?:\.\d+)?)\s*(?:km|kilometers?|miles?)/i,  // "radius of 5km"
      /within\s*(\d+(?:\.\d+)?)\s*(?:km|kilometers?|miles?)/i,  // "within 10km"
      /range\s*(?:of|at)?\s*(\d+(?:\.\d+)?)\s*(?:km|kilometers?|miles?)/i,  // "range of 5km"
      /(\d+(?:\.\d+)?)\s*(?:km|kilometers?|miles?)\s*(?:radius|range)/i  // "5km radius" or "3 miles radius"
    ];
    
    for (const pattern of radiusPatterns) {
      const radiusMatch = text.match(pattern);
      if (radiusMatch) {
        // Extract the number and unit from the match
        const fullMatch = radiusMatch[0];
        const numberMatch = radiusMatch[1] || fullMatch.match(/(\d+(?:\.\d+)?)/)?.[1];
        
        if (numberMatch) {
          // Build a string with number and unit for parseDistance
          const unitMatch = fullMatch.match(/(?:km|kilometer|mile|miles|m|meter)/i);
          const unit = unitMatch ? unitMatch[0] : 'km';
          const distanceStr = `${numberMatch} ${unit}`;
          
          parameters.radius_km = this.parseDistance(distanceStr);
          if (parameters.radius_km > 0) {
            break;
          }
        }
      }
    }

    // Radar avoidance directive
    if (lowerText.includes("avoid radar") || lowerText.includes("radar avoid")) {
      parameters.avoid_radar = true;
    }
    
    return {
      action,
      target,
      parameters
    };
  }

  private parseDistance(distanceStr: string): number {
    // Enhanced distance parsing
    const match = distanceStr.match(/(\d+(?:\.\d+)?)/);
    if (match) {
      const value = parseFloat(match[1]);
      const unit = distanceStr.toLowerCase();
      if (unit.includes("km") || unit.includes("kilometer")) {
        return value;
      } else if (unit.includes("mile") && !unit.includes("nautical")) {
        // Handle both "mile" and "miles"
        return value * 1.60934; // Convert to km
      } else if (unit.includes("nautical") || unit.includes("nm")) {
        return value * 1.852; // Nautical miles to km
      } else if (unit.includes("meter") || unit.includes(" m ") || unit.match(/\d+\s*m\b/)) {
        return value / 1000; // Convert to km
      }
    }
    return 0;
  }

  private parseDuration(durationStr: string): number {
    // Parse time duration to seconds
    const match = durationStr.match(/(\d+)/);
    if (match) {
      const value = parseInt(match[1]);
      const unit = durationStr.toLowerCase();
      if (unit.includes("hour") || unit.includes("hr")) {
        return value * 3600;
      } else if (unit.includes("minute") || unit.includes("min")) {
        return value * 60;
      } else if (unit.includes("second") || unit.includes("sec")) {
        return value;
      }
    }
    return 0;
  }

  private parseSpeed(speedStr: string): number {
    // Parse speed to m/s
    const match = speedStr.match(/(\d+)/);
    if (match) {
      const value = parseFloat(match[1]);
      const unit = speedStr.toLowerCase();
      if (unit.includes("m/s") || unit.includes("ms")) {
        return value;
      } else if (unit.includes("km/h") || unit.includes("kmh")) {
        return value / 3.6; // km/h to m/s
      } else if (unit.includes("mph")) {
        return value * 0.44704; // mph to m/s
      } else if (unit.includes("knot") || unit.includes("kt")) {
        return value * 0.514444; // knots to m/s
      }
    }
    return 0;
  }

  private parseAltitude(altitudeStr: string): number {
    // Parse altitude to meters
    const match = altitudeStr.match(/(\d+)/);
    if (match) {
      const value = parseFloat(match[1]);
      const unit = altitudeStr.toLowerCase();
      if (unit.includes("meter") || unit.includes(" m ") || unit.match(/\d+\s*m\b/)) {
        return value;
      } else if (unit.includes("feet") || unit.includes("ft")) {
        return value * 0.3048; // feet to meters
      }
    }
    return 0;
  }

  private determineIntentType(intent: StructuredIntent): "task" | "query" | "constraint" | "override" {
    // Simple heuristics
    if (!intent || !intent.action) {
      return "task"; // Default if no action
    }
    const queryActions = ["status", "query", "what", "where"];
    if (queryActions.some(q => intent.action.toLowerCase().includes(q))) {
      return "query";
    }
    return "task"; // Default
  }
}
