import { type LayerDefinition, type LayerRegistry, type LayerData } from "../node_gen/BaseClass";
/**
 * This module defines an empty LAYER_REGISTRY to act as a placeholder
 */
// 1. Define the Registry Object (Empty initially)
export const LAYER_REGISTRY: LayerRegistry = {};

// 2. Helper to populate it safely
export function registerLayer(key: string, cls: LayerDefinition<LayerData>) {
    LAYER_REGISTRY[key] = cls;
}
