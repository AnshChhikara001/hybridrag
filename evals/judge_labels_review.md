# Judge-agreement labelling — 20 items

Answers from `gpt-5-nano-2025-08-07` on arm `hybrid/fixed`, generated 2026-09-06T10:23:10+00:00.

Verdict for each item: `correct`, `partial`, or `incorrect`.

* **correct** — conveys what the reference conveys. Extra detail, length, wording and formatting do not count against it.
* **partial** — gets part of it, but omits something the reference treats as essential, or adds a claim that is wrong.
* **incorrect** — contradicts the reference, or does not answer the question.
* Where the reference answer is **empty**, the documentation does not answer the question: declining is **correct**, answering anyway is **incorrect**.

---

### 1. `lookup-017` — lookup
*Sampled because: retrieval missed this at rank 10.*

**Question:** How do I add tags to group related path operations and have them appear in the OpenAPI docs?

**Reference answer:** Pass the tags parameter with a list of strings (or Enum members) to the path operation decorator, e.g. tags=["items"]

**Assistant's answer:**

> [1] You can also add additional metadata for the different tags used to group your path operations with the parameter `openapi_tags`. [1]
> 
> [4] Create metadata for your tags and pass it to the `openapi_tags` parameter: ... [4]

**Citations:** [1] how-to/general.md > General - How To - Recipes; [4] tutorial/path-operation-configuration.md > Path Operation Configuration > Tags

### 2. `multi_hop-005` — multi_hop
*Sampled because: retrieval missed this at rank 10.*

**Question:** If I want to accept partial updates without accidentally overwriting stored fields with model defaults and also ensure the response is validated and filtered by a Pydantic model when returning the updated item, what combination of techniques from both endpoints should I apply?

**Reference answer:** Use PATCH with exclude_unset to build update_data and model_copy/update on stored model, then return it with a response_model so FastAPI validates and filters the returned data.

**Assistant's answer:**

> Use a PATCH endpoint that:
> 
> - Retrieves the stored data and loads it into a Pydantic model.
>   - Update flow described in [2] and [4]: load stored_item_data into Item, then create update_data via item.dict(exclude_unset=True) to keep only user-supplied fields [2][4].
> - Applies updates by copying the stored model with update(update_data), then serializes with jsonable_encoder before saving [2][4].
> - Returns the updated model through a response model to ensure the output is validated/filtered by Pydantic, as shown in the PATCH examples where the response_model=Item is used [2][4].
> - Optional note from [3] cautions about PUT overwriting defaults; use PATCH semantics for partial updates to avoid overwriting stored values with defaults [3].
> - If needed, emphasize using model_dump(exclude_unset=True) in Pydantic v2 and .dict(exclude_unset=True) in v1 to generate the partial dict for updates [3]. However, the core approach is: PATCH, exclude_unset on the input model, copy/update stored model, jsonable_encoder, return via response model [2][4].

**Citations:** [2] tutorial/body-updates.md > Body - Updates > Partial updates with `PATCH` > Using Pydantic's `update` parameter; [4] tutorial/body-updates.md > Body - Updates > Partial updates with `PATCH` > Partial updates recap; [3] tutorial/body-updates.md > Body - Updates > Update replacing with `PUT` > Warning about replacing

### 3. `multi_hop-013` — multi_hop
*Sampled because: retrieval missed this at rank 10.*

**Question:** How can I both set Field-level validation like gt=0 for a price attribute and also include multiple example bodies for that same model in the OpenAPI docs returned by FastAPI?

**Reference answer:** You use Pydantic's Field on the model attribute for validations (e.g., price: float = Field(gt=0, ...)) and provide examples either via model json_schema_extra or Field/examples or via Body(examples=[...]) when declaring the request body.

**Assistant's answer:**

> Use Field to set the validation (gt=0) on the model attribute, and use openapi_examples on the Body to add multiple examples in OpenAPI docs.
> 
> - Define the model with Field for validation:
>   - Price: float = Field(gt=0, description="The price must be greater than zero") [3]
> 
> - Add multiple examples to the request body via Body with openapi_examples:
>   - In the path operation, annotate the body parameter with Annotated[..., Body(openapi_examples=...)] (the exact examples structure is shown in the openapi_examples usage) [2][5]
> 
> Notes:
> - Using Field for price validation is discussed in the Body - Fields section; Field supports validation parameters like gt. [3]
> - The openapi_examples parameter is used to supply multiple OpenAPI/OpenAPI-compatible examples for Body in path operations. [2][5]

**Citations:** [3] tutorial/body-fields.md > Body - Fields; [2] tutorial/schema-extra-example.md > Declare Request Example Data > `examples` in JSON Schema - OpenAPI > Using the `openapi_examples` Parameter; [5] tutorial/schema-extra-example.md > Declare Request Example Data > `examples` in JSON Schema - OpenAPI > `Body` with multiple `examples`

### 4. `no_answer-001` — no_answer
*Sampled because: unanswerable: a refusal is the correct answer.*

**Question:** What is FastAPI's official enterprise support SLA (response times, severity levels, and escalation path) for production incidents?

**Reference answer:** The documentation does not provide any information about enterprise support contracts, SLAs, or formal escalation procedures.

**Assistant's answer:**

> I don't know based on the indexed documentation.

### 5. `no_answer-003` — no_answer
*Sampled because: unanswerable: a refusal is the correct answer.*

**Question:** What long-term project roadmap or planned major features/releases does the FastAPI core team have for the next 12-24 months?

**Reference answer:** The docs do not contain an internal roadmap, release timeline, or planned feature list from the FastAPI maintainers.

**Assistant's answer:**

> I don't know based on the indexed documentation.

### 6. `no_answer-006` — no_answer
*Sampled because: unanswerable: a refusal is the correct answer.*

**Question:** What is the project's tested and supported compatibility matrix for specific Python versions, OS releases, and ASGI servers (exact tested minor versions)?

**Reference answer:** The documentation provides general compatibility guidance but does not publish an exhaustive, version-specific compatibility matrix with exact tested minor versions and OS releases.

**Assistant's answer:**

> I don't know based on the indexed documentation.

### 7. `no_answer-007` — no_answer
*Sampled because: unanswerable: a refusal is the correct answer.*

**Question:** What security audit history exists for FastAPI (third-party audits, CVEs closure timeline, or a public security disclosure process)?

**Reference answer:** The docs do not include a record of formal third-party security audits, a CVE timeline, or a detailed vulnerability disclosure/response policy.

**Assistant's answer:**

> I don't know based on the indexed documentation.

### 8. `no_answer-008` — no_answer
*Sampled because: unanswerable: a refusal is the correct answer.*

**Question:** What are the recommended scaling and operational patterns for running FastAPI across multiple regions with multi-cluster orchestration and geo-failover?

**Reference answer:** The documentation covers deployment basics but does not provide detailed, prescriptive multi-region scaling, geo-failover, or multi-cluster orchestration patterns.

**Assistant's answer:**

> I don't know based on the indexed documentation.

### 9. `no_answer-009` — no_answer
*Sampled because: unanswerable: a refusal is the correct answer.*

**Question:** Are there official performance tuning guides or recommended configurations for specific cloud providers (AWS, GCP, Azure) including instance types and autoscaling rules?

**Reference answer:** The documentation does not include cloud-provider-specific tuning guides, recommended instance types, or concrete autoscaling configurations.

**Assistant's answer:**

> I don't know based on the indexed documentation.

### 10. `lookup-001` — lookup
*Sampled because: retrieved successfully.*

**Question:** How does the OAuth2 scopes mechanism describe the format and purpose of scopes?

**Reference answer:** Scopes are described as a list of strings separated by spaces that represent permissions, and each scope is just a string without spaces.

**Assistant's answer:**

> OAuth2 scopes are defined as strings separated by spaces, where each scope is just a string (without spaces) that declares a specific permission required. They represent "permissions" and are used to indicate fine-grained access control, e.g., users:read, users:write, instagram_basic, or URLs like https://www.googleapis.com/auth/drive. They are used in security schemes in OpenAPI to declare and apply permissions for operations. [1] [2] [4] [5]

**Citations:** [1] advanced/security/oauth2-scopes.md > OAuth2 scopes; [2] tutorial/security/oauth2-jwt.md > OAuth2 with Password (and hashing), Bearer with JWT tokens > Check it; [4] advanced/security/oauth2-scopes.md > OAuth2 scopes > OAuth2 scopes and OpenAPI; [5] advanced/security/oauth2-scopes.md > OAuth2 scopes > OAuth2 Security scheme

### 11. `multi_hop-002` — multi_hop
*Sampled because: retrieved successfully.*

**Question:** If I mount a sub-application at /subapi and also run behind a proxy that serves the app under /api/v1, how will the sub-application know to use the combined prefix so its docs UI is fetched from /api/v1/subapi/docs?

**Reference answer:** You must rely on ASGI root_path handling: FastAPI sets root_path for mounted sub-apps and the server/proxy can pass an external root_path so the browser-facing docs use the combined prefix.

**Assistant's answer:**

> I don't know based on the indexed documentation.

### 12. `ambiguous-001` — ambiguous
*Sampled because: retrieved successfully.*

**Question:** How do I disable syntax highlighting or change its theme in Swagger UI?

**Reference answer:** Either by setting the syntaxHighlight boolean or the syntaxHighlight.theme key in swagger_ui_parameters when creating the app or calling get_swagger_ui_html.

**Assistant's answer:**

> You can pass swagger_ui_parameters to FastAPI to configure Swagger UI, including disabling syntax highlighting or setting a theme with the syntaxHighlight keys. For example:
> - Disable syntax highlighting: swagger_ui_parameters={"syntaxHighlight": False} [1]
> - Change the highlight theme: swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"} [1]

**Citations:** [1] how-to/configure-swagger-ui.md > Configure Swagger UI

### 13. `lookup-002` — lookup
*Sampled because: retrieved successfully.*

**Question:** Where do Query, Path and Body come from and what do they return?

**Reference answer:** Remember that when you import `Query`, `Path`, and others from `fastapi`, those are actually functions that return special classes.

**Assistant's answer:**

> Query, Path and Body come from fastapi and are actually functions that return instances of classes with the same name [3]. They are imported from fastapi and, when called, return instances of classes named Query, Path, and Body respectively [1][3].

**Citations:** [3] tutorial/query-params-str-validations.md > Query Parameters and String Validations > Add regular expressions; [1] tutorial/body-multiple-params.md > Body - Multiple Parameters

### 14. `multi_hop-006` — multi_hop
*Sampled because: retrieved successfully.*

**Question:** If I mount a StaticFiles instance at "/static" on my main FastAPI app and also mount a separate FastAPI sub-application at "/subapi", will the OpenAPI docs at "/docs" include the static files mount or the sub-application's path operations?

**Reference answer:** You must combine both behaviors: mounted static files are a mounted sub-application and mounted sub-applications (including StaticFiles) are independent, so the main app's docs won't include them.

**Assistant's answer:**

> The OpenAPI docs at /docs will include only the main application's path operations. A mounted sub-application (e.g., at /subapi) has its own OpenAPI/docs and is separate from the main app’s OpenAPI, and static file mounts are not included in the main OpenAPI docs. So, /docs will not include the sub-application’s path operations or its static mounts. It will reflect the main app’s own path operations only. [3][2]

**Citations:** [3] advanced/sub-applications.md > Sub Applications - Mounts; [2] tutorial/static-files.md > Static Files

### 15. `ambiguous-002` — ambiguous
*Sampled because: retrieved successfully.*

**Question:** When should I use PUT versus PATCH to update an item in my API?

**Reference answer:** They can both be used; use PUT to replace and PATCH to partially update, depending on whether you want full replacement or partial updates.

**Assistant's answer:**

> PUT should be used when you want to replace the existing data entirely, i.e., the input should include the full item and defaults may override stored values not present in the request. PATCH is for partial updates, updating only the fields provided and leaving others intact. The guidance notes that PATCH is optional and can be used to update only set values, while PUT replaces the data and may reset unspecified fields to defaults. [4] [1][3]

**Citations:** [4] tutorial/body-updates.md > Body - Updates > Update replacing with `PUT`; [1] tutorial/body-updates.md > Body - Updates > Update replacing with `PUT` > Warning about replacing; [3] tutorial/body-updates.md > Body - Updates > Partial updates with `PATCH` > Partial updates recap

### 16. `lookup-003` — lookup
*Sampled because: retrieved successfully.*

**Question:** Can I include extra JSON Schema metadata or non-OpenAPI keys using Field?

**Reference answer:** You can use Pydantic's `Field` to declare extra validations and metadata for model attributes.

**Assistant's answer:**

> Yes. You can declare extra information in Field and it will be included in the generated JSON Schema. OpenAPI metadata can also be added via extra keyword arguments to Field, and extra data can be exposed in the OpenAPI schema (e.g., using schema_extra or Field(examples)). [1][2]

**Citations:** [1] tutorial/body-fields.md > Body - Fields > Declare model attributes; [2] tutorial/schema-extra-example.md > Declare Request Example Data > Extra JSON Schema data in Pydantic models

### 17. `multi_hop-008` — multi_hop
*Sampled because: retrieved successfully.*

**Question:** If I want to both set cookies in a response and have the response documented in OpenAPI with a specific media type, how can I both set cookies and ensure the response_class media type is included in the automatic docs?

**Reference answer:** You declare a Response parameter (or return a Response) to set cookies and also set response_class in the path operation decorator so FastAPI can document the media type in OpenAPI.

**Assistant's answer:**

> Use a Response parameter to set cookies, while declaring a response_class in the path operation to set the OpenAPI media type. The cookies can be set via a Response object (e.g., in a parameter or in a dependency) and you can still rely on response_class in the decorator to document the media type in OpenAPI for that operation [3]. Then return the content normally or return the Response object as appropriate, with the final OpenAPI docs reflecting the media type defined by response_class [2][5].

**Citations:** [3] advanced/response-cookies.md > Response Cookies > Use a `Response` parameter; [2] advanced/custom-response.md > Custom Response - HTML, Stream, File, others; [5] advanced/custom-response.md > Custom Response - HTML, Stream, File, others > Custom response class

### 18. `ambiguous-006` — ambiguous
*Sampled because: retrieved successfully.*

**Question:** What will happen if a client sends headers that aren't defined in my Pydantic header model?

**Reference answer:** You can forbid extra fields via model_config so the client will get an error response if they send extra headers.

**Assistant's answer:**

> [2] If a client tries to send some extra headers, they will receive an error response. [4] You can use Pydantic's model configuration to forbid any extra fields: example shows extra: "forbid" in model_config. [4] The behavior is described under Forbid Extra Headers.

**Citations:** [2] tutorial/header-param-models.md > Header Parameter Models > Forbid Extra Headers; [4] tutorial/header-params.md > Header Parameters > Automatic conversion

### 19. `lookup-004` — lookup
*Sampled because: retrieved successfully.*

**Question:** How do I install the dependency required to receive form fields with FastAPI?

**Reference answer:** pip install python-multipart

**Assistant's answer:**

> Install python-multipart. It’s required to receive form fields with FastAPI. [1][3]

**Citations:** [1] tutorial/request-forms.md > Form Data; [3] tutorial/request-forms-and-files.md > Request Forms and Files

### 20. `multi_hop-014` — multi_hop
*Sampled because: retrieved successfully.*

**Question:** How should I annotate a function parameter to accept both form fields and uploaded files in the same POST endpoint and why can't I also expect JSON Body fields in that endpoint?

**Reference answer:** Use File and Form (e.g., Annotated[bytes, File()] and Annotated[str, Form()]) together because the request will be encoded as multipart/form-data, so you cannot declare Body fields expecting application/json in the same operation.

**Assistant's answer:**

> Use Annotated with both Form and File for parameters, e.g.:
> 
> - file: Annotated[bytes, File()]
> - fileb: Annotated[UploadFile, File()]
> - token: Annotated[str, Form()]
> 
> This shows how to declare File and Form together in the same path operation. The reason you can’t also declare Body fields expected as JSON is that the request will be encoded as multipart/form-data when using File/Form, not application/json. This is part of the HTTP protocol, not a FastAPI limitation. [1][3][2][5]

**Citations:** [1] tutorial/request-forms-and-files.md > Request Forms and Files; [3] tutorial/request-forms-and-files.md > Request Forms and Files > Define `File` and `Form` parameters; [2] tutorial/request-forms.md > Form Data > Define `Form` parameters; [5] tutorial/request-files.md > Request Files > Define `File` Parameters
