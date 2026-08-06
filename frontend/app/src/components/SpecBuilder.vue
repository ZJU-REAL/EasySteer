<script setup lang="ts">
/** The full SteeringSpec form: vector list + conflict policy + debug. */
import { useI18n } from "../i18n";
import {
  CONFLICT_POLICIES,
  cloneSpec,
  defaultVectorSpec,
  type SteeringSpec,
} from "../lib/spec";
import VectorSpecEditor from "./VectorSpecEditor.vue";

const props = defineProps<{ spec: SteeringSpec }>();
const { t } = useI18n();

function addVector(): void {
  props.spec.vectors.push(defaultVectorSpec());
}

function removeVector(index: number): void {
  props.spec.vectors.splice(index, 1);
}

function duplicateVector(index: number): void {
  const copy = cloneSpec({ vectors: [props.spec.vectors[index]], conflict: "priority", debug: false });
  props.spec.vectors.splice(index + 1, 0, copy.vectors[0]);
}
</script>

<template>
  <div class="spec-builder">
    <div class="field-row top-row">
      <div class="field">
        <label>{{ t("conflict_label") }}</label>
        <select v-model="spec.conflict" class="mono full">
          <option v-for="policy in CONFLICT_POLICIES" :key="policy" :value="policy">
            {{ t(`conflict_${policy}` as any) }}
          </option>
        </select>
      </div>
      <div class="field debug-field">
        <label class="inline-check">
          <input v-model="spec.debug" type="checkbox" />
          {{ t("debug_label") }}
        </label>
      </div>
    </div>

    <div class="vector-list">
      <VectorSpecEditor
        v-for="(vector, i) in spec.vectors"
        :key="i"
        :vector="vector"
        :index="i"
        :removable="spec.vectors.length > 1"
        @remove="removeVector(i)"
        @duplicate="duplicateVector(i)"
      />
    </div>

    <button class="small add-btn" @click="addVector">+ {{ t("add_vector_btn") }}</button>
  </div>
</template>

<style scoped>
.top-row {
  align-items: end;
}

.debug-field {
  flex: 0 0 auto !important;
  padding-bottom: 6px;
}

.vector-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.add-btn {
  margin-top: 8px;
}
</style>
