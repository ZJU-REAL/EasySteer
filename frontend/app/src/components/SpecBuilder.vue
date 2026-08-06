<script setup lang="ts">
/**
 * The SteeringSpec form. Vectors live behind a tab bar (one editor at a
 * time) so the panel keeps a steady height no matter how many vectors a
 * spec carries.
 */
import { computed, ref, watch } from "vue";
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

const active = ref(0);

const count = computed(() => props.spec.vectors.length);

// A spec replaced from JSON or a gallery preset can have fewer vectors.
watch(count, (n) => {
  if (active.value > n - 1) active.value = Math.max(0, n - 1);
});

function addVector(): void {
  props.spec.vectors.push(defaultVectorSpec());
  active.value = count.value - 1;
}

function removeActive(): void {
  props.spec.vectors.splice(active.value, 1);
  active.value = Math.min(active.value, count.value - 1);
}

function duplicateActive(): void {
  const copy = cloneSpec({
    vectors: [props.spec.vectors[active.value]],
    conflict: "priority",
  });
  props.spec.vectors.splice(active.value + 1, 0, copy.vectors[0]);
  active.value += 1;
}
</script>

<template>
  <div class="spec-builder">
    <div class="builder-toolbar">
      <div class="tab-bar">
        <button
          v-for="(vector, i) in spec.vectors"
          :key="i"
          class="tab"
          :class="{ active: active === i }"
          @click="active = i"
        >
          {{ t("vector_n_title", { n: i + 1 }) }}
          <span v-if="vector.algorithm !== 'direct'" class="tab-algo mono">{{
            vector.algorithm
          }}</span>
        </button>
        <button class="tab add-tab" :title="t('add_vector_btn')" @click="addVector">＋</button>
      </div>
      <span class="spacer"></span>
      <button class="small" @click="duplicateActive">{{ t("duplicate_vector_btn") }}</button>
      <button class="small" :disabled="count < 2" @click="removeActive">
        {{ t("remove_btn") }}
      </button>
    </div>

    <VectorSpecEditor :key="active" :vector="spec.vectors[active]" />

    <div class="field-row spec-footer">
      <div class="field">
        <label>{{ t("conflict_label") }}</label>
        <select v-model="spec.conflict" class="mono full" :disabled="count < 2">
          <option v-for="policy in CONFLICT_POLICIES" :key="policy" :value="policy">
            {{ t(`conflict_${policy}` as any) }}
          </option>
        </select>
      </div>
    </div>
  </div>
</template>

<style scoped>
.builder-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding-bottom: 12px;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--border);
}

.tab-algo {
  font-size: 10.5px;
  opacity: 0.75;
}

.add-tab {
  padding: 5px 10px;
  font-size: 14px;
  line-height: 1;
}

.spec-footer {
  align-items: flex-end;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}

.spec-footer .field {
  margin-bottom: 0;
  flex: 0 1 340px;
}

</style>
